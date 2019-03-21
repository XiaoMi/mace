mace {
  global:
    *GPUContextBuilder*;
    *MaceEngineConfig*;
    *MaceTensor*;
    *MaceEngine*;
    *CreateMaceEngineFromProto*;
    *GetBigLittleCoreIDs*;
    *MaceVersion*;
    *GetCapability*;

    # api for static library of models
    *mace*port**;
    *mace*MaceStatus*;
    *mace*NetDef*;
    *mace*MemoryType*;
    *mace*DataType*;
    *mace*InputInfo*;
    *mace*OutputInfo*;
    *mace*OutputShape*;
    *mace*OperatorDef*;
    *mace*ConstTensor*;
    *mace*Argument*;
    *mace*Split*;
    *mace*MemoryBlock*;
    *google*protobuf*;

  local:
    *;
};
