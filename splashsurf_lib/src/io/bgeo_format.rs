//! Helper functions for the BGEO file format

use crate::io::io_utils::IteratorExt;
use crate::mesh::{AttributeData, MeshAttribute};
use crate::{Real, RealConvert, io::io_utils};
use anyhow::{Context, anyhow};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use nalgebra::Vector3;
use nom::{Finish, Parser};
use num_traits::{FromPrimitive, ToPrimitive};
use parser::bgeo_parser;
use std::fs::{File, OpenOptions};
use std::io;
use std::io::{BufWriter, Read};
use std::path::Path;

// TODO: Find out why there is a 1.0 float value between position vector and id int in splishsplash output
// TODO: Better error messages, skip nom errors

/// Convenience function for loading particles from a BGEO file path
pub fn particles_from_bgeo<R: Real, P: AsRef<Path>>(
    bgeo_file: P,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    // Load positions from BGEO file
    let bgeo_file = load_bgeo_file(bgeo_file).context("Error while loading BGEO file")?;
    particles_from_bgeo_file(&bgeo_file)
}

/// Returns particle positions from a loaded BGEO file
pub fn particles_from_bgeo_file<R: Real>(
    bgeo_file: &BgeoFile,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    let position_storage = {
        let storage = &bgeo_file.positions;

        if let AttributeStorage::Vector(dim, storage) = storage {
            assert_eq!(*dim, 3);
            assert_eq!(storage.len() % dim, 0);
            storage
        } else {
            return Err(anyhow!("Positions are not stored as vectors in BGEO file"));
        }
    };

    // Convert the array storage into individual vectors
    let positions: Vec<_> =
        io_utils::try_convert_scalar_slice_to_vectors(position_storage, |v| v.try_convert())?;
    Ok(positions)
}

/// Convenience function that converts the point attributes with the given names from the BGEO file into mesh attributes
pub fn attributes_from_bgeo_file<R: Real>(
    bgeo_file: &BgeoFile,
    names: &[String],
) -> Result<Vec<MeshAttribute<R>>, anyhow::Error> {
    let mut mesh_attributes = Vec::new();

    'fields: for field_name in names {
        for (name, storage) in bgeo_file.attribute_data.iter() {
            if name == field_name {
                let attribute_data = storage
                    .try_into_attribute_data::<R>()
                    .context(anyhow!("Failed to convert attribute \"{name}\""))?;
                mesh_attributes.push(MeshAttribute::new(field_name.clone(), attribute_data));
                continue 'fields;
            }
        }
    }

    Ok(mesh_attributes)
}

/// Loads and parses a BGEO file to memory
pub fn load_bgeo_file<P: AsRef<Path>>(bgeo_file: P) -> Result<BgeoFile, anyhow::Error> {
    let mut buf = Vec::new();
    let mut is_compressed = false;

    // First check if the file is gzip compressed
    {
        let file = File::open(bgeo_file.as_ref()).context("Unable to open file for reading")?;
        let mut gz = GzDecoder::new(file);
        if gz.header().is_some() {
            is_compressed = true;
            gz.read_to_end(&mut buf)
                .context("Error during gzip decompression")?;
        }
    }

    // Otherwise just read the raw file
    if !is_compressed {
        // File has to be opened again because the Gz header check already reads parts of the file
        let mut file = File::open(bgeo_file).context("Unable to open file for reading")?;
        file.read_to_end(&mut buf)
            .context("Error while loading the file content")?;
    }

    let (_, file) = bgeo_parser()
        .parse(&buf[..])
        .finish()
        .map_err(|err| err.into_anyhow())
        .context("Error while parsing the BGEO file contents")?;

    Ok(file)
}

pub fn particles_to_bgeo<R: Real, P: AsRef<Path>>(
    particles: &[Vector3<R>],
    bgeo_file: P,
    enable_compression: bool,
) -> Result<(), anyhow::Error> {
    let path = bgeo_file.as_ref();
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .context("Cannot open file for writing JSON")?;
    let writer = BufWriter::new(file);

    let bgeo = particles_to_bgeo_impl(particles)?;
    write_bgeo_file(&bgeo, writer, enable_compression)
}

fn particles_to_bgeo_impl<R: Real>(particles: &[Vector3<R>]) -> Result<BgeoFile, anyhow::Error> {
    let particles_f32 = particles
        .iter()
        .flat_map(|x| x.as_slice())
        .copied()
        .map(|x| Some(x.to_f32())?)
        .map(|vec| {
            vec.ok_or_else(|| {
                anyhow!(
                    "Failed to convert coordinate from input float type to f32, value out of range?"
                )
            })
        })
        .try_collect_with_capacity(particles.len())?;

    Ok(BgeoFile {
        header: BgeoHeader {
            magic_bytes: [66, 103, 101, 111],
            version_char: 86,
            version: 5,
            num_points: particles.len().to_i32().ok_or_else(|| {
                anyhow!(
                    "number of particles ({}) is too large for bgeo format (max {})",
                    particles.len(),
                    i32::MAX
                )
            })?,
            num_prims: 0,
            num_point_groups: 0,
            num_prim_groups: 0,
            num_point_attrib: 0,
            num_vertex_attrib: 0,
            num_prim_attrib: 0,
            num_attrib: 0,
        },
        positions: AttributeStorage::Vector(3, particles_f32),
        weights: AttributeStorage::Float(vec![1.0; particles.len()]),
        attribute_definitions: Vec::new(),
        attribute_data: Vec::new(),
    })
}

pub fn write_bgeo_file<W: io::Write>(
    bgeo: &BgeoFile,
    writer: W,
    enable_compression: bool,
) -> Result<(), anyhow::Error> {
    fn write_bgeo<W2: io::Write>(bgeo: &BgeoFile, mut writer: W2) -> Result<(), anyhow::Error> {
        writer.write_all(&bgeo.header.magic_bytes)?;
        writer.write_all(&bgeo.header.version_char.to_be_bytes())?;
        writer.write_all(&bgeo.header.version.to_be_bytes())?;

        writer.write_all(&bgeo.header.num_points.to_be_bytes())?;
        writer.write_all(&bgeo.header.num_prims.to_be_bytes())?;
        writer.write_all(&bgeo.header.num_point_groups.to_be_bytes())?;
        writer.write_all(&bgeo.header.num_prim_groups.to_be_bytes())?;
        writer.write_all(&bgeo.header.num_point_attrib.to_be_bytes())?;
        writer.write_all(&bgeo.header.num_vertex_attrib.to_be_bytes())?;
        writer.write_all(&bgeo.header.num_prim_attrib.to_be_bytes())?;
        writer.write_all(&bgeo.header.num_attrib.to_be_bytes())?;

        // Write attribute definitions
        for attrib in &bgeo.attribute_definitions {
            writer.write_all(&(attrib.name.len() as u16).to_be_bytes())?;
            writer.write_all(attrib.name.as_bytes())?;
            writer.write_all(&(attrib.size as u16).to_be_bytes())?;
            writer.write_all(&(attrib.attr_type.to_i32()).to_be_bytes())?;
            for &default_val in &attrib.default_values {
                writer.write_all(&(default_val.to_be_bytes()))?;
            }
        }

        let special_attribs = [&bgeo.positions, &bgeo.weights];
        let attrib_iter = || {
            special_attribs
                .iter()
                .copied()
                .chain(bgeo.attribute_data.iter().map(|(_, s)| s))
        };

        let num_points = attrib_iter().map(|s| s.num_points()).max().unwrap();
        for i in 0..num_points {
            for attrib in attrib_iter() {
                // TODO: Use default values
                match attrib {
                    AttributeStorage::Int(v) => {
                        writer.write_all(&(v[i].to_be_bytes()))?;
                    }
                    AttributeStorage::Float(v) => {
                        writer.write_all(&(v[i].to_be_bytes()))?;
                    }
                    AttributeStorage::Vector(n, v) => {
                        for j in 0..*n {
                            writer.write_all(&(v[i * n + j].to_be_bytes()))?;
                        }
                    }
                }
            }
        }

        // End bytes
        writer.write_all(&0x00_u8.to_be_bytes())?;
        writer.write_all(&0xff_u8.to_be_bytes())?;

        Ok(())
    }

    if enable_compression {
        write_bgeo(bgeo, GzEncoder::new(writer, Compression::fast()))
    } else {
        write_bgeo(bgeo, writer)
    }
}

/// Struct representing a parsed BGEO file
#[derive(Clone, Debug)]
pub struct BgeoFile {
    /// Header data of the BGEO file
    pub header: BgeoHeader,
    /// Positions of all points
    pub positions: AttributeStorage,
    /// An unknown float attribute of all points
    pub weights: AttributeStorage,
    /// Definitions of all remaining point attributes
    pub attribute_definitions: Vec<AttribDefinition>,
    /// Data of all remaining point attributes
    pub attribute_data: Vec<(String, AttributeStorage)>,
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

    fn to_i32(self) -> i32 {
        match self {
            BgeoAttributeType::Float => 0,
            BgeoAttributeType::Int => 1,
            BgeoAttributeType::String => 2,
            BgeoAttributeType::IndexedString => 4,
            BgeoAttributeType::Vector => 5,
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

impl AttributeStorage {
    /// Returns the number of points for this attribute storage
    fn num_points(&self) -> usize {
        match self {
            AttributeStorage::Int(v) => v.len(),
            AttributeStorage::Float(v) => v.len(),
            AttributeStorage::Vector(n, v) => v.len() / n,
        }
    }

    /// Tries to convert this BGEO attribute storage into a mesh [`AttributeData`] storage
    fn try_into_attribute_data<R: Real>(&self) -> Result<AttributeData<R>, anyhow::Error> {
        match self {
            AttributeStorage::Int(data) => io_utils::try_convert_scalar_slice(data, u64::from_i32)
                .map(|v| AttributeData::ScalarU64(v))
                .context(anyhow!("failed to convert integer attribute")),
            AttributeStorage::Float(data) => io_utils::try_convert_scalar_slice(data, R::from_f32)
                .map(|v| AttributeData::ScalarReal(v))
                .context(anyhow!("failed to convert float attribute")),
            AttributeStorage::Vector(n, data) => {
                if *n == 3 {
                    io_utils::try_convert_scalar_slice_to_vectors(data, R::from_f32)
                        .map(|v| AttributeData::Vector3Real(v))
                        .context(anyhow!("failed to convert vector attribute"))
                } else {
                    Err(anyhow!("unsupported vector attribute size: {}", n))
                }
            }
        }
    }
}

/// Parsers used to parse the BGEO format
mod parser {
    use nom::branch::alt;
    use nom::bytes::streaming::{tag, take};
    use nom::combinator::{map_opt, map_res, not, recognize, value, verify};
    use nom::multi::{count, fold};
    use nom::number::complete as number;
    use nom::{IResult, Parser};

    use super::error::{BgeoParserError, BgeoParserErrorKind, bgeo_error, make_bgeo_error};
    use super::{AttribDefinition, AttributeStorage, BgeoAttributeType, BgeoFile, BgeoHeader};

    pub fn bgeo_parser<'a>()
    -> impl Parser<&'a [u8], Output = BgeoFile, Error = BgeoParserError<&'a [u8]>> {
        move |input: &'a [u8]| -> IResult<&'a [u8], BgeoFile, BgeoParserError<&'a [u8]>> {
            // Parse file header and attribute definitions
            let (input, header) = parse_header(input)?;
            let (input, named_attribute_definitions) =
                count(parse_attr_def, header.num_point_attrib as usize).parse(input)?;

            // Add the "position" attribute which should always be present
            let special_attribute_definitions = {
                let mut special_attribute_definitions = Vec::new();
                special_attribute_definitions.push(AttribDefinition {
                    name: String::from("position"),
                    size: 3,
                    attr_type: BgeoAttributeType::Vector,
                    default_values: vec![0, 0, 0],
                });
                // TODO: This additional float value appears between positions and ids in splishsplash BGEO files
                //  Not sure what this is exactly
                special_attribute_definitions.push(AttribDefinition {
                    name: String::from("unknown"),
                    size: 1,
                    attr_type: BgeoAttributeType::Float,
                    default_values: vec![0],
                });
                special_attribute_definitions
            };

            // Parse the point attribute data
            let (input, (mut special_attribute_data, attribute_data)) = parse_points(
                input,
                header.num_points as usize,
                special_attribute_definitions.as_slice(),
                named_attribute_definitions.as_slice(),
            )?;

            assert_eq!(special_attribute_data.len(), 2);

            let weights = special_attribute_data.pop().unwrap();
            let positions = special_attribute_data.pop().unwrap();

            let file = BgeoFile {
                header,
                positions,
                weights,
                attribute_definitions: named_attribute_definitions,
                attribute_data,
            };

            Ok((input, file))
        }
    }

    /// Parses the BGEO format magic bytes
    fn parse_magic_bytes(input: &[u8]) -> IResult<&[u8], &[u8], BgeoParserError<&[u8]>> {
        tag("Bgeo").parse(input)
    }

    /// Parses the (unsupported) new BGEO format magic bytes
    fn parse_new_magic_bytes(input: &[u8]) -> IResult<&[u8], &[u8], BgeoParserError<&[u8]>> {
        tag([0x7f, 0x4e, 0x53, 0x4a].as_slice()).parse(input)
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
        ))
        .parse(input)
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
        let (input, name_length) = number::be_u16(input)?;
        let (input, name) = map_res(take(name_length as usize), |input: &[u8]| {
            std::str::from_utf8(input).map_err(|_| BgeoParserErrorKind::InvalidAttributeName)
        })
        .parse(input)?;

        let (input, size) = number::be_u16(input)?;
        let (input, attr_type) = bgeo_error(
            BgeoParserErrorKind::UnknownAttributeType,
            map_opt(number::be_i32, BgeoAttributeType::from_i32),
        )(input)?;

        match attr_type {
            BgeoAttributeType::Float | BgeoAttributeType::Int | BgeoAttributeType::Vector => {
                let (input, default_values) = count(number::be_i32, size as usize).parse(input)?;

                let attr = AttribDefinition {
                    name: name.to_string(),
                    size: size as usize,
                    attr_type,
                    default_values,
                };

                Ok((input, attr))
            }
            _ => Err(make_bgeo_error(
                input,
                BgeoParserErrorKind::UnsupportedAttributeType(attr_type),
            )),
        }
    }

    /// Parses all attribute values for points
    fn parse_points<'a>(
        input: &'a [u8],
        num_points: usize,
        special_attribs: &[AttribDefinition],
        named_attribs: &[AttribDefinition],
    ) -> IResult<
        &'a [u8],
        (Vec<AttributeStorage>, Vec<(String, AttributeStorage)>),
        BgeoParserError<&'a [u8]>,
    > {
        // Construct a parser for each attribute
        let mut parsers: Vec<_> = special_attribs
            .iter()
            .chain(named_attribs.iter())
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
            //let mut parser_funs: Vec<_> = parsers.iter_mut().map(|p| p.parser()).collect();
            // For each point...
            for _ in 0..num_points {
                // ...apply all parsers in succession
                for parser in parsers.iter_mut() {
                    let (i, _) = parser.parse(input)?;
                    input = i;
                }
            }

            input
        };

        // Collect the individual attribute storages
        let mut special_attrib_data = Vec::new();
        let mut named_attrib_data = Vec::new();

        for parser in parsers.into_iter() {
            assert_eq!(
                num_points * parser.attrib.size,
                parser.storage.len(),
                "failed to read attribute \"{}\" (type {:?}): number of read attribute values ({}) does not match expected number of attribute values ({} = {} points * {} attribute components)",
                parser.attrib.name,
                parser.attrib.attr_type,
                parser.storage.len(),
                num_points * parser.attrib.size,
                num_points,
                parser.attrib.size
            );
            if special_attrib_data.len() < special_attribs.len() {
                special_attrib_data.push(parser.storage);
            } else {
                named_attrib_data.push((parser.attrib.name, parser.storage));
            }
        }

        Ok((input, (special_attrib_data, named_attrib_data)))
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

        fn parse<'a>(
            &mut self,
            input: &'a [u8],
        ) -> IResult<&'a [u8], (), BgeoParserError<&'a [u8]>> {
            match self.attrib.attr_type {
                BgeoAttributeType::Int => {
                    let storage = self.storage.as_int_mut();
                    map_res(number::be_i32, |val| {
                        storage.push(val);
                        Ok(())
                    })
                    .parse(input)
                }
                BgeoAttributeType::Float => {
                    let storage = self.storage.as_float_mut();
                    map_res(number::be_f32, |val| {
                        storage.push(val);
                        Ok(())
                    })
                    .parse(input)
                }
                BgeoAttributeType::Vector => {
                    let storage = self.storage.as_vec_mut();
                    let size = self.attrib.size;
                    fold(
                        size,
                        number::be_f32,
                        || (),
                        |_, val| {
                            storage.push(val);
                        },
                    )
                    .parse(input)
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
            matches!(self, BgeoParserErrorKind::NomError(_))
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
                err
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
        F: Parser<I, Output = O, Error = BgeoParserError<I>>,
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

#[test]
fn test_bgeo_read_dam_break() {
    let input_file = Path::new("../data/dam_break_frame_9_6859_particles.bgeo");
    let particles = particles_from_bgeo::<f32, _>(input_file).unwrap();

    assert_eq!(particles.len(), 6859);

    use crate::Aabb3d;
    let aabb = Aabb3d::from_points(&particles);
    let enclosing = Aabb3d::new(
        Vector3::new(-2.0, 0.03, -0.8),
        Vector3::new(-0.3, 0.7, 0.72),
    );

    assert!(enclosing.contains_aabb(&aabb));
}

#[test]
fn test_bgeo_read_dam_break_attributes() {
    let input_file = Path::new("../data/dam_break_frame_9_6859_particles.bgeo");

    let bgeo_file = load_bgeo_file(input_file).unwrap();

    let particles = particles_from_bgeo_file::<f32>(&bgeo_file).unwrap();
    assert_eq!(particles.len(), 6859);

    assert_eq!(bgeo_file.attribute_definitions.len(), 3);
    assert_eq!(bgeo_file.attribute_data.len(), 3);

    use crate::Aabb3d;
    let aabb = Aabb3d::from_points(&particles);
    let enclosing = Aabb3d::new(
        Vector3::new(-2.0, 0.03, -0.8),
        Vector3::new(-0.3, 0.7, 0.72),
    );

    assert!(enclosing.contains_aabb(&aabb));

    let attribs = attributes_from_bgeo_file::<f32>(
        &bgeo_file,
        &[
            "id".to_string(),
            "density".to_string(),
            "velocity".to_string(),
        ],
    )
    .unwrap();

    assert_eq!(attribs.len(), 3);

    assert!(matches!(attribs[0].data, AttributeData::ScalarU64(_)));
    assert!(matches!(attribs[1].data, AttributeData::ScalarReal(_)));
    assert!(matches!(attribs[2].data, AttributeData::Vector3Real(_)));

    if let AttributeData::ScalarU64(ids) = &attribs[0].data {
        assert_eq!(ids.len(), particles.len());
        assert_eq!(ids[0], 30);
        assert_eq!(ids[1], 11);
        assert_eq!(ids[2], 12);
    }

    if let AttributeData::ScalarReal(densities) = &attribs[1].data {
        assert_eq!(densities.len(), particles.len());
        assert_eq!(densities[0], 1000.1286);
        assert_eq!(densities[1], 1001.53424);
        assert_eq!(densities[2], 1001.6626);
    }

    if let AttributeData::Vector3Real(velocities) = &attribs[2].data {
        assert_eq!(velocities.len(), particles.len());
        assert_eq!(
            velocities[0],
            Vector3::new(0.3670507, -0.41762838, 0.42659923)
        );
    }
}

#[test]
fn test_bgeo_write_dam_break() {
    let input_file = Path::new("../data/dam_break_frame_9_6859_particles.bgeo");
    let particles = particles_from_bgeo::<f32, _>(input_file).unwrap();

    assert_eq!(particles.len(), 6859);

    let bgeo_to_write = particles_to_bgeo_impl(&particles).unwrap();

    let mut buffer: Vec<u8> = Vec::new();
    write_bgeo_file(&bgeo_to_write, &mut buffer, false).unwrap();

    let (_, bgeo_read) = bgeo_parser()
        .parse(buffer.as_slice())
        .finish()
        .map_err(|err| err.into_anyhow())
        .context("Error while parsing the BGEO file contents")
        .unwrap();

    let particles_read = particles_from_bgeo_file(&bgeo_read).unwrap();

    assert_eq!(particles_read.len(), 6859);
    assert_eq!(particles, particles_read);
}

#[test]
fn test_bgeo_roundtrip_uncompressed() {
    let input_file = Path::new("../data/dam_break_frame_9_6859_particles.bgeo");
    let bgeo = load_bgeo_file(input_file).unwrap();

    let mut orig = Vec::new();
    let mut is_compressed = false;

    // First check if the file is gzip compressed
    {
        let file = File::open(input_file)
            .context("Unable to open file for reading")
            .unwrap();
        let mut gz = GzDecoder::new(file);
        if gz.header().is_some() {
            is_compressed = true;
            // Read if compressed
            gz.read_to_end(&mut orig)
                .context("Error during gzip decompression")
                .unwrap();
        }
    }

    // Read if not compressed
    if !is_compressed {
        File::open(input_file)
            .unwrap()
            .read_to_end(&mut orig)
            .unwrap();
    }

    let mut buffer: Vec<u8> = Vec::new();
    write_bgeo_file(&bgeo, &mut buffer, false).unwrap();

    assert_eq!(orig.len(), buffer.len());
    assert_eq!(&orig[0..buffer.len()], buffer.as_slice());
}
