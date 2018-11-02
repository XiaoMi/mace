mace {
  global:
    *GPUContextBuilder*;
    *MaceEngineConfig*;
    *MaceTensor*;
    *MaceEngine*;
    *CreateMaceEngineFromProto*;
    *GetBigLittleCoreIDs*;
    *MaceVersion*;

    # api for static library of models
    *mace*logging*LogMessage*;
    *mace*MaceStatus*;
    *mace*NetDef*;
    *mace*MemoryType*;
    *mace*DataType*;
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
