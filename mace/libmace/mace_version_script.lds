mace {
  global:
    *MaceTensor*;
    *MaceEngine*;
    *CreateMaceEngineFromProto*;
    *FileStorageFactory*;
    *SetKVStorageFactory*;
    *SetOpenCLBinaryPaths*;
    *SetOpenCLParameterPath*;
    *SetGPUHints*;
    *SetOpenMPThreadPolicy*;
    *SetOpenMPThreadAffinity*;
    *GetBigLittleCoreIDs*;
    *MaceVersion*;

    # api for static library of models
    *mace*logging*LogMessage*;
    *mace*NetDef*;
    *mace*MemoryArena*;
    *mace*InputInfo*;
    *mace*OutputInfo*;
    *mace*OutputShape*;
    *mace*OperatorDef*;
    *mace*ConstTensor*;
    *mace*Argument*;
    *mace*MemoryBlock*;
    *google*protobuf*;

  local:
    *;
};
