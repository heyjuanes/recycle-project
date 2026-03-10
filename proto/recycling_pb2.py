# -*- coding: utf-8 -*-
# Este archivo fue generado automaticamente por grpc_tools.protoc
# No modificar manualmente. Para regenerar ejecutar:
# python -m grpc_tools.protoc -I proto --python_out=proto --grpc_python_out=proto proto/recycling.proto
# source: recycling.proto
# Protobuf Python Version: 6.31.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    31,
    1,
    '',
    'recycling.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0frecycling.proto\x12\trecycling\"&\n\x10\x44\x65tectionRequest\x12\x12\n\nimage_data\x18\x01 \x01(\x0c\"z\n\x0e\x44\x65tectedObject\x12\x12\n\nclass_name\x18\x01 \x01(\t\x12\x12\n\nconfidence\x18\x02 \x01(\x02\x12\n\n\x02x1\x18\x03 \x01(\x02\x12\n\n\x02y1\x18\x04 \x01(\x02\x12\n\n\x02x2\x18\x05 \x01(\x02\x12\n\n\x02y2\x18\x06 \x01(\x02\x12\x10\n\x08material\x18\x07 \x01(\t\"a\n\x11\x44\x65tectionResponse\x12*\n\x07objects\x18\x01 \x03(\x0b\x32\x19.recycling.DetectedObject\x12\x0f\n\x07success\x18\x02 \x01(\x08\x12\x0f\n\x07message\x18\x03 \x01(\t2`\n\x12RecyclingInference\x12J\n\rDetectObjects\x12\x1b.recycling.DetectionRequest\x1a\x1c.recycling.DetectionResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'recycling_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_DETECTIONREQUEST']._serialized_start=30
  _globals['_DETECTIONREQUEST']._serialized_end=68
  _globals['_DETECTEDOBJECT']._serialized_start=70
  _globals['_DETECTEDOBJECT']._serialized_end=192
  _globals['_DETECTIONRESPONSE']._serialized_start=194
  _globals['_DETECTIONRESPONSE']._serialized_end=291
  _globals['_RECYCLINGINFERENCE']._serialized_start=293
  _globals['_RECYCLINGINFERENCE']._serialized_end=389
# @@protoc_insertion_point(module_scope)
