use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::Context;
use flate2::read::GzDecoder;
use nom::{Finish, Parser};
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::Real;

use parser::bgeo_parser;

// TODO: Find out why there is a 1.0 float value between position vector and id int in Splishsplash output
// TODO: Better error messages, skip nom errors

pub fn particles_from_bgeo<R: Real, P: AsRef<Path>>(
    bgeo_file: P,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    // Load positions from BGEO file
    let position_storage = {
        let mut bgeo_file = load_bgeo_file(bgeo_file).context("Error while loading BGEO file")?;

        println!("header: {:?}", bgeo_file.header);
        println!("attrs: {:?}", bgeo_file.point_attributes);

        let storage = bgeo_file
            .points
            .remove("position")
            .expect("Positions should always be in BGEO file");

        if let AttributeStorage::Vector(dim, storage) = storage {
            assert_eq!(dim, 3);
            assert_eq!(storage.len() % dim, 0);
            storage
        } else {
            panic!("Positions are not stored as vectors");
        }
    };

    // Convert the array storage into individual vectors
    let positions: Vec<_> = position_storage
        .chunks(3)
        .map(|p| {
            Vector3::new(
                R::from_f32(p[0]).unwrap(),
                R::from_f32(p[1]).unwrap(),
                R::from_f32(p[2]).unwrap(),
            )
        })
        .collect();

    Ok(positions)
}

/// Loads and parses a BGEO file to memory
pub fn load_bgeo_file<P: AsRef<Path>>(bgeo_file: P) -> Result<BgeoFile, anyhow::Error> {
    let file = File::open(bgeo_file).context("Unable to open file for reading")?;

    let mut gz = GzDecoder::new(file);
    let mut buf = Vec::new();
    gz.read_to_end(&mut buf)
        .context("Error during gzip decompression")?;

    //buf[3] = b'V';
    println!("{}", String::from_utf8_lossy(&buf[0..100]));

    let (_, file) = bgeo_parser()
        .parse(&buf[..])
        .finish()
        .map_err(|err| err.into_anyhow())
        .context("Error while parsing the BGEO file contents")?;

    Ok(file)
}

/// Struct representing a parsed BGEO file
#[derive(Clone, Debug)]
pub struct BgeoFile {
    pub header: BgeoHeader,
    pub point_attributes: Vec<AttribDefinition>,
    pub points: HashMap<String, AttributeStorage>,
}

/// The header data of a BGEO file
#[derive(Clone, Debug)]
pub struct BgeoHeader {
    pub magic_bytes: [u8; 4],
    pub version_char: u8,
    pub version: i32,
    pub num_points: i32,
    pub num_prims: i32,
    pub num_point_groups: i32,
    pub num_prim_groups: i32,
    pub num_point_attrib: i32,
    pub num_vertex_attrib: i32,
    pub num_prim_attrib: i32,
    pub num_attrib: i32,
}

/// The type of a BGEO attribute
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BgeoAttributeType {
    Float,
    Int,
    String,
    IndexedString,
    Vector,
}

impl BgeoAttributeType {
    fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(BgeoAttributeType::Float),
            1 => Some(BgeoAttributeType::Int),
            2 => Some(BgeoAttributeType::String),
            4 => Some(BgeoAttributeType::IndexedString),
            5 => Some(BgeoAttributeType::Vector),
            _ => None,
        }
    }
}

/// Definition of a BGEO attribute
#[derive(Clone, Debug)]
pub struct AttribDefinition {
    pub name: String,
    pub size: usize,
    pub attr_type: BgeoAttributeType,
    pub default_values: Vec<i32>,
}

/// Storage for BGEO entity attributes
#[derive(Clone, Debug)]
pub enum AttributeStorage {
    Int(Vec<i32>),
    Float(Vec<f32>),
    Vector(usize, Vec<f32>),
}

/// Parsers used to parse the BGEO format
mod parser {
    use std::collections::HashMap;

    use nom::branch::alt;
    use nom::bytes::streaming::{tag, take};
    use nom::combinator::{map, map_opt, map_res, not, recognize, value, verify};
    use nom::multi::count;
    use nom::number::complete as number;
    use nom::{IResult, Parser};

    use super::error::{bgeo_error, make_bgeo_error, BgeoParserError, BgeoParserErrorKind};
    use super::{AttribDefinition, AttributeStorage, BgeoAttributeType, BgeoFile, BgeoHeader};

    pub fn bgeo_parser<'a>() -> impl Parser<&'a [u8], BgeoFile, BgeoParserError<&'a [u8]>> {
        move |input: &'a [u8]| -> IResult<&'a [u8], BgeoFile, BgeoParserError<&'a [u8]>> {
            // Parse file header and attribute definitions
            let (input, header) = parse_header(input)?;
            let (input, point_attributes) =
                count(parse_attr_def, header.num_point_attrib as usize)(input)?;

            // Add the "position" attribute which should always be present
            let point_attributes = {
                let mut point_attributes = point_attributes;
                point_attributes.insert(
                    0,
                    AttribDefinition {
                        name: String::from("position"),
                        size: 3,
                        attr_type: BgeoAttributeType::Vector,
                        default_values: vec![0, 0, 0],
                    },
                );
                // TODO: This additional float value appears between positions and ids in splishsplash BGEO files
                //  Not sure what this is exactly
                point_attributes.insert(
                    1,
                    AttribDefinition {
                        name: String::from("density"),
                        size: 1,
                        attr_type: BgeoAttributeType::Float,
                        default_values: vec![0],
                    },
                );
                point_attributes
            };

            // Parse the point attribute data
            let (input, point_data) = parse_points(
                input,
                header.num_points as usize,
                point_attributes.as_slice(),
            )?;

            let file = BgeoFile {
                header,
                point_attributes,
                points: point_data,
            };

            Ok((input, file))
        }
    }

    /// Parses the BGEO format magic bytes
    fn parse_magic_bytes(input: &[u8]) -> IResult<&[u8], &[u8], BgeoParserError<&[u8]>> {
        tag("Bgeo")(input)
    }

    /// Parses the (unsupported) new BGEO format magic bytes
    fn parse_new_magic_bytes(input: &[u8]) -> IResult<&[u8], &[u8], BgeoParserError<&[u8]>> {
        tag([0x7f, 0x4e, 0x53, 0x4a])(input)
    }

    /// Parses and validates the magic bytes of a BGEO file header
    fn parse_header_magic(input: &[u8]) -> IResult<&[u8], &[u8], BgeoParserError<&[u8]>> {
        alt((
            // First, try to parse the supported magic number
            parse_magic_bytes,
            // Otherwise, if also the magic number of the new BGEO format is not found,
            // this is a completely unknown format
            bgeo_error(
                BgeoParserErrorKind::MagicBytesNotFound,
                recognize(parse_new_magic_bytes),
            )
            // But also recognizing the new BGEO format magic number should result in an error
            .and_then(bgeo_error(
                BgeoParserErrorKind::UnsupportedFormatVersion,
                value("".as_bytes(), not(parse_new_magic_bytes)),
            )),
        ))(input)
    }

    #[test]
    fn test_bgeo_header_magic_parser() {
        use nom::Err;

        assert_eq!(
            parse_header_magic(b"Bgeooo"),
            Ok(("oo".as_bytes(), "Bgeo".as_bytes()))
        );

        if let Err::Error(e) = parse_header_magic(&[0x7f, 0x4e, 0x53, 0x4a]).unwrap_err() {
            assert_eq!(
                e.first_bgeo_error().unwrap(),
                BgeoParserErrorKind::UnsupportedFormatVersion
            );
        } else {
            panic!();
        }

        if let Err::Error(e) = parse_header_magic(b"Hgeo").unwrap_err() {
            assert_eq!(
                e.first_bgeo_error().unwrap(),
                BgeoParserErrorKind::MagicBytesNotFound
            );
        } else {
            panic!();
        }
    }

    /// Parses the version numbers of a BGEO file header
    fn parse_version(input: &[u8]) -> IResult<&[u8], (u8, i32), BgeoParserError<&[u8]>> {
        let (input, version_char) = number::be_u8(input)?;
        let (input, version) = bgeo_error(
            BgeoParserErrorKind::UnsupportedFormatVersion,
            verify(number::be_i32, |v: &i32| *v == 5),
        )(input)?;
        Ok((input, (version_char, version)))
    }

    /// Parses the full BGEO file header
    fn parse_header(input: &[u8]) -> IResult<&[u8], BgeoHeader, BgeoParserError<&[u8]>> {
        let (input, magic_bytes) = parse_header_magic(input)?;
        let (input, (version_char, version)) = parse_version(input)?;

        let (input, num_points) = number::be_i32(input)?;
        let (input, num_prims) = number::be_i32(input)?;
        let (input, num_point_groups) = number::be_i32(input)?;

        let (input, num_prim_groups) = number::be_i32(input)?;
        let (input, num_point_attrib) = number::be_i32(input)?;
        let (input, num_vertex_attrib) = number::be_i32(input)?;
        let (input, num_prim_attrib) = number::be_i32(input)?;
        let (input, num_attrib) = number::be_i32(input)?;

        let header = BgeoHeader {
            magic_bytes: [
                magic_bytes[0],
                magic_bytes[1],
                magic_bytes[2],
                magic_bytes[3],
            ],
            version_char,
            version,
            num_points,
            num_prims,
            num_point_groups,
            num_prim_groups,
            num_point_attrib,
            num_vertex_attrib,
            num_prim_attrib,
            num_attrib,
        };

        // TODO: Validate header, e.g. values >= 0

        Ok((input, header))
    }

    /// Parses a single BGEO attribute definition
    fn parse_attr_def(input: &[u8]) -> IResult<&[u8], AttribDefinition, BgeoParserError<&[u8]>> {
        let input = input;
        let (input, name_length) = number::be_u16(input)?;
        let (input, name) = map_res(take(name_length as usize), |input: &[u8]| {
            std::str::from_utf8(input).map_err(|_| BgeoParserErrorKind::InvalidAttributeName)
        })(input)?;

        let (input, size) = number::be_u16(input)?;
        let (input, attr_type) = bgeo_error(
            BgeoParserErrorKind::UnknownAttributeType,
            map_opt(number::be_i32, BgeoAttributeType::from_i32),
        )(input)?;

        match attr_type {
            BgeoAttributeType::Float | BgeoAttributeType::Int | BgeoAttributeType::Vector => {
                let (input, default_values) = count(number::be_i32, size as usize)(input)?;

                let attr = AttribDefinition {
                    name: name.to_string(),
                    size: size as usize,
                    attr_type,
                    default_values,
                };

                Ok((input, attr))
            }
            _ => {
                return Err(make_bgeo_error(
                    input,
                    BgeoParserErrorKind::UnsupportedAttributeType(attr_type),
                ));
            }
        }
    }

    /// Parses all attribute values for points
    fn parse_points<'a>(
        input: &'a [u8],
        num_points: usize,
        attribs: &[AttribDefinition],
    ) -> IResult<&'a [u8], HashMap<String, AttributeStorage>, BgeoParserError<&'a [u8]>> {
        // Construct a parser for each attribute
        let mut parsers: Vec<_> = attribs
            .iter()
            .cloned()
            .map(|attrib| {
                // Allocate storage for the attribute
                let storage = AttributeStorage::with_capacity(num_points, &attrib)
                    .expect("Unimplemented attribute storage");
                AttributeParser::new(attrib, storage)
            })
            .collect();

        // Run the parsers alternating
        let input = {
            let mut input = input;

            // Get the parser functions
            let mut parser_funs: Vec<_> = parsers.iter_mut().map(|p| p.parser()).collect();
            // For each point...
            for _ in 0..num_points {
                // ...apply all parsers in succession
                for parser in &mut parser_funs {
                    let (i, _) = (parser).parse(input)?;
                    input = i;
                }
            }

            input
        };

        // Collect the individual attribute storages into a hashmap
        let mut attrib_data = HashMap::new();
        for parser in parsers.into_iter() {
            assert_eq!(num_points * parser.attrib.size, parser.storage.len());
            attrib_data.insert(parser.attrib.name, parser.storage);
        }

        Ok((input, attrib_data))
    }

    impl AttributeStorage {
        /// Pre-allocates storage for the given attribute definition
        fn with_capacity(num_entities: usize, attrib: &AttribDefinition) -> Option<Self> {
            let capacity = num_entities * attrib.size;
            match attrib.attr_type {
                BgeoAttributeType::Int => Some(AttributeStorage::Int(Vec::with_capacity(capacity))),
                BgeoAttributeType::Float => {
                    Some(AttributeStorage::Float(Vec::with_capacity(capacity)))
                }
                BgeoAttributeType::Vector => Some(AttributeStorage::Vector(
                    attrib.size,
                    Vec::with_capacity(capacity),
                )),
                _ => None,
            }
        }

        /// Returns the number of scalar values in the storage
        pub fn len(&self) -> usize {
            match self {
                AttributeStorage::Int(v) => v.len(),
                AttributeStorage::Float(v) => v.len(),
                AttributeStorage::Vector(_, v) => v.len(),
            }
        }

        /// Returns a mutable reference to the Int storage, panics if it is not the correct variant
        pub fn as_int_mut(&mut self) -> &mut Vec<i32> {
            if let AttributeStorage::Int(v) = self {
                v
            } else {
                panic!("Storage is not an Int storage");
            }
        }

        /// Returns a mutable reference to the Float storage, panics if it is not the correct variant
        pub fn as_float_mut(&mut self) -> &mut Vec<f32> {
            if let AttributeStorage::Float(v) = self {
                v
            } else {
                panic!("Storage is not an Float storage");
            }
        }

        /// Returns a mutable reference to the Vector storage, panics if it is not the correct variant
        pub fn as_vec_mut(&mut self) -> &mut Vec<f32> {
            if let AttributeStorage::Vector(_, v) = self {
                v
            } else {
                panic!("Storage is not an Vector storage");
            }
        }
    }

    /// Helper struct for parsing BGEO attributes
    struct AttributeParser {
        attrib: AttribDefinition,
        storage: AttributeStorage,
    }

    impl AttributeParser {
        fn new(attrib: AttribDefinition, storage: AttributeStorage) -> Self {
            Self { attrib, storage }
        }

        /// Returns a boxed parser for this attribute
        fn parser<'a, 'b>(
            &'b mut self,
        ) -> Box<dyn Parser<&'a [u8], (), BgeoParserError<&'a [u8]>> + 'b> {
            match self.attrib.attr_type {
                BgeoAttributeType::Int => {
                    let storage = self.storage.as_int_mut();
                    Box::new(
                        move |input: &'a [u8]| -> IResult<&'a [u8], (), BgeoParserError<&'a [u8]>> {
                            let (input, value) = number::be_i32(input)?;
                            //println!("i32: {}", value);
                            storage.push(value);
                            Ok((input, ()))
                        },
                    )
                }
                BgeoAttributeType::Float => {
                    let storage = self.storage.as_float_mut();
                    Box::new(
                        move |input: &'a [u8]| -> IResult<&'a [u8], (), BgeoParserError<&'a [u8]>> {
                            let (input, value) = number::be_f32(input)?;
                            //println!("f32: {}", value);
                            storage.push(value);
                            Ok((input, ()))
                        },
                    )
                }
                BgeoAttributeType::Vector => {
                    let storage = self.storage.as_vec_mut();
                    let size = self.attrib.size;
                    Box::new(
                        move |input: &'a [u8]| -> IResult<&'a [u8], (), BgeoParserError<&'a [u8]>> {
                            let (input, _) = count(
                                map(number::be_f32, |val| {
                                    //println!("vec: {}", val);
                                    storage.push(val);
                                }),
                                size,
                            )(input)?;
                            Ok((input, ()))
                        },
                    )
                }
                _ => panic!("Unsupported attribute type"),
            }
        }
    }
}

/// Error types and traits used by the BGEO parser
mod error {
    use super::BgeoAttributeType;
    use anyhow::anyhow;
    use nom::error::{ContextError, ErrorKind, FromExternalError, ParseError};
    use nom::{IResult, Parser};
    use std::borrow::Cow;

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub enum BgeoParserErrorKind {
        MagicBytesNotFound,
        UnsupportedFormatVersion,
        InvalidAttributeName,
        UnsupportedAttributeType(BgeoAttributeType),
        UnknownAttributeType,
        Context(Cow<'static, str>),
        NomError(ErrorKind),
    }

    impl BgeoParserErrorKind {
        /// Returns whether the variant is an internal nom error
        pub fn is_nom_error(&self) -> bool {
            match self {
                BgeoParserErrorKind::NomError(_) => true,
                _ => false,
            }
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    pub struct BgeoParserError<I> {
        pub backtrace: Vec<(I, BgeoParserErrorKind)>,
    }

    #[allow(dead_code)]
    impl<I> BgeoParserError<I> {
        /// Construct a new error with the given input and error kind
        pub fn from_error_kind(input: I, kind: BgeoParserErrorKind) -> Self {
            Self {
                backtrace: vec![(input, kind)],
            }
        }

        /// Appends an error to the backtrace with the given input and error kind and returns self
        pub fn with_append(mut self, input: I, kind: BgeoParserErrorKind) -> Self {
            self.backtrace.push((input, kind));
            self
        }

        /// Appends a context message to the backtrace and return self
        pub fn with_context<S: Into<Cow<'static, str>>>(self, input: I, ctx: S) -> Self {
            self.with_append(input, BgeoParserErrorKind::Context(ctx.into()))
        }

        /// Wraps the error into a (recoverable) nom::Err::Error
        pub fn into_nom_error(self) -> nom::Err<Self> {
            nom::Err::Error(self)
        }

        /// Iterator that skips all errors in the beginning of the backtrace that are not actual BGEO format errors (i.e. internal nom parser errors)
        pub fn begin_bgeo_errors(&self) -> impl Iterator<Item = &(I, BgeoParserErrorKind)> {
            self.backtrace.iter().skip_while(|(_, e)| e.is_nom_error())
        }

        /// Returns the kind of the first error in the backtrace that is an actual BGEO format error kind (i.e. skips internal nom parser errors)
        pub fn first_bgeo_error(&self) -> Option<BgeoParserErrorKind> {
            self.begin_bgeo_errors().next().map(|(_, ek)| ek).cloned()
        }

        /// Drops all inputs from the error backtrace, resulting in a trace of only the error [BgeoParserErrorKind] entries
        pub fn into_error_kinds(self) -> Vec<BgeoParserErrorKind> {
            self.backtrace.into_iter().map(|(_, e)| e).collect()
        }

        /// Convert this error into an anyhow error
        pub fn into_anyhow(self) -> anyhow::Error {
            if let Some((_, first)) = self.backtrace.first() {
                let mut err = anyhow!("{:?}", first);
                for (_, kind) in self.backtrace.iter().skip(1) {
                    err = err.context(format!("{:?}", kind));
                }
                return err;
            } else {
                anyhow!("Unknown")
            }
        }
    }

    impl<I> ParseError<I> for BgeoParserError<I> {
        fn from_error_kind(input: I, kind: ErrorKind) -> Self {
            Self::from_error_kind(input, BgeoParserErrorKind::NomError(kind))
        }

        fn append(input: I, kind: ErrorKind, other: Self) -> Self {
            other.with_append(input, BgeoParserErrorKind::NomError(kind))
        }
    }

    impl<I> ContextError<I> for BgeoParserError<I> {
        fn add_context(input: I, ctx: &'static str, other: Self) -> Self {
            other.with_context(input, Cow::Borrowed(ctx))
        }
    }

    impl<I> FromExternalError<I, BgeoParserErrorKind> for BgeoParserError<I> {
        fn from_external_error(input: I, _kind: ErrorKind, e: BgeoParserErrorKind) -> Self {
            Self::from_error_kind(input, e)
        }
    }

    /// Creates a new [BgeoParserError] wrapped into a [nom::Err::Error] variant
    pub fn make_bgeo_error<I>(input: I, kind: BgeoParserErrorKind) -> nom::Err<BgeoParserError<I>> {
        BgeoParserError::from_error_kind(input, kind).into_nom_error()
    }

    /// Parser that attaches an error of the given [BgeoParserErrorKind] if the inner parser fails
    pub fn bgeo_error<I: Clone, F, O>(
        kind: BgeoParserErrorKind,
        mut f: F,
    ) -> impl FnMut(I) -> IResult<I, O, BgeoParserError<I>>
    where
        F: Parser<I, O, BgeoParserError<I>>,
    {
        use nom::Err;
        move |i: I| match f.parse(i.clone()) {
            Ok(o) => Ok(o),
            Err(Err::Incomplete(i)) => Err(Err::Incomplete(i)),
            Err(Err::Error(e)) => Err(Err::Error(e.with_append(i, kind.clone()))),
            Err(Err::Failure(e)) => Err(Err::Failure(e.with_append(i, kind.clone()))),
        }
    }
}
