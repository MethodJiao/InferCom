#pragma once

#define DefineSuper(B)  public: typedef B T_Super; 

//schema
#define PBM_SCHEMA_Demo_W            L"DemoSchema_leuJjR"
#define PBM_SCHEMA_Demo              "DemoSchema_leuJjR"

//element
#define PBM_CLASS_CUBE_Demo_W            L"CubeDemo"
#define PBM_CLASS_CUBE_Demo              "CubeDemo"
#define PBM_CLASS_SOLID_Demo_W            L"SolidDemo"
#define PBM_CLASS_SOLID_Demo              "SolidDemo"
#define PBM_CLASS_OPENNING_Demo              "OpenningDemo"
#define PBM_RELSHIP_CUBEWITHOPENNING         "CubeWithOpenning"
#define PBM_CLASS_CROSSARM_Demo         "CrossArmDemo"
#define PBM_CLASS_TOWER_Demo            "TowerDemo"
#define PBM_CLASS_DATAMANAGER_Demo_W            L"DataManagerDemo"
#define PBM_CLASS_DATAMANAGER_Demo              "DataManagerDemo"

#define PBM_CLASS_INDEPENDENT_BRIDGE   "IndependentBridge"
#define PBM_CLASS_INDEPENDENT_BRIDGE_W   L"IndependentBridge"

#define PBM_CLASS_TUBE_Demo_W            L"Tube"
#define PBM_CLASS_TUBE_Demo              "Tube"

#define PBM_CLASS_UNIVERSAL_BEAM_Demo_W            L"UniversalBeamDemo"
#define PBM_CLASS_UNIVERSAL_BEAM_Demo              "UniversalBeamDemo"

#define PBM_CLASS_INSULATOR_Demo_W      L"InsulatorDemo"
#define PBM_CLASS_INSULATOR_Demo        "InsulatorDemo"

#define PBM_CLASS_ARCH_WALL_Demo_W            L"ArchDemo"
#define PBM_CLASS_ARCH_WALL_Demo              "ArchDemo"

#define Demo_CREATE(_name_)\
    public:\
    static _name_##P create(){ return new _name_(); }\
    static _name_##P create(BIMBase::Core::BPDataCR data)\
{\
    _name_##P pElement = new _name_(); \
    if(SUCCESS==pElement->initFromData(data)) return pElement; \
    return NULL;\
};

#define Demo_EXTENSION(_name_)\
class _name_##Extension : public IBPObjectExtension\
{\
protected:\
    virtual IBPObjectPtr        _getBPObject() override\
			    {\
    return _name_::create(); \
			    } \
    virtual IBPObjectPtr        _getBPObject(BIMBase::Core::BPDataCR instance) override\
			    {\
    return _name_::create(instance); \
			    }\
};