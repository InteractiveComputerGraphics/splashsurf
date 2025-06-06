# The contents of this file are partially copied from the blender sequence loader (https://github.com/InteractiveComputerGraphics/blender-sequence-loader)

try:
    import gzip
    import numpy as np
    import meshio


    def readbgeo_to_meshio(filepath):
        with gzip.open(filepath, 'r') as file:
            byte = file.read(5)
            if byte != b"BgeoV":
                raise Exception('not bgeo file format')
            byte = file.read(4)
            version = int.from_bytes(byte, byteorder="big")
            if version != 5:
                raise Exception('bgeo file not version 5')

            header = {}
            point_attributes = {}
            point_attributes_names = []
            point_attributes_sizes = []
            point_attributes_types = []
            position = None

            byte = file.read(4)
            header['nPoints'] = int.from_bytes(byte, byteorder="big")

            byte = file.read(4)
            header['nPrims'] = int.from_bytes(byte, byteorder="big")

            byte = file.read(4)
            header['nPointGroups'] = int.from_bytes(byte, byteorder="big")

            byte = file.read(4)
            header['nPrimGroups'] = int.from_bytes(byte, byteorder="big")

            byte = file.read(4)
            header['nPointAttrib'] = int.from_bytes(byte, byteorder="big")

            byte = file.read(4)
            header['nVertexAttrib'] = int.from_bytes(byte, byteorder="big")

            byte = file.read(4)
            header['nPrimAttrib'] = int.from_bytes(byte, byteorder="big")

            byte = file.read(4)
            header['nAttrib'] = int.from_bytes(byte, byteorder="big")

            particle_size = 4

            for _ in range(header['nPointAttrib']):
                byte = file.read(2)
                namelength = int.from_bytes(byte, byteorder="big")
                name_binary = file.read(namelength)
                name = name_binary.decode('utf-8')
                point_attributes_names.append(name)

                byte = file.read(2)
                size = int.from_bytes(byte, byteorder="big")
                point_attributes_sizes.append(size)
                particle_size += size

                byte = file.read(4)
                input_dtype = int.from_bytes(byte, byteorder="big")
                if input_dtype == 0:
                    point_attributes_types.append('FLOAT')
                    # read default value
                    # not going to do anything about it
                    byte = file.read(size * 4)
                elif input_dtype == 1:
                    point_attributes_types.append('INT')
                    # read default value
                    # not going to do anything about it
                    byte = file.read(size * 4)
                elif input_dtype == 5:
                    point_attributes_types.append('VECTOR')
                    # read default value
                    # not going to do anything about it
                    byte = file.read(size * 4)
                else:
                    raise Exception('input_dtype unknown/unsupported')
            byte = file.read(particle_size * header['nPoints'] * 4)
            # > means big endian
            attribute_data = np.frombuffer(byte, dtype='>f')
            attribute_data = np.reshape(attribute_data, (header['nPoints'], particle_size))
            # the first 3 column is its position data
            position = attribute_data[:, :3]
            # the 4th column is homogeneous coordiante, which is all 1, and will be ignored

            current_attribute_start_point = 4
            for i in range(header['nPointAttrib']):
                if point_attributes_types[i] == 'FLOAT':
                    point_attributes[point_attributes_names[i]] = attribute_data[:, current_attribute_start_point]
                    current_attribute_start_point += 1
                elif point_attributes_types[i] == 'VECTOR':
                    point_attributes[
                        point_attributes_names[i]] = attribute_data[:,
                                                                    current_attribute_start_point:current_attribute_start_point + 3]
                    current_attribute_start_point += 3
                elif point_attributes_types[i] == 'INT':
                    data = (attribute_data[:, current_attribute_start_point]).tobytes()
                    # > means big endian
                    point_attributes[point_attributes_names[i]] = np.frombuffer(data, dtype='>i')
                    current_attribute_start_point += 1
            remaining = file.read()
            if not remaining == b'\x00\xff':
                raise Exception("file didn't end")
            return meshio.Mesh(position, [('vertex', [])], point_data=point_attributes)


    # no need for write function
    meshio.register_format("bgeo", [".bgeo"], readbgeo_to_meshio, {".bgeo": None})

except ImportError:
    pass