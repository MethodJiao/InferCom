#include "stdafx.h"



using namespace DemoObject;

void ToolExampleTubeRotate::Rotate(std::vector<::BIMBase::Core::BPEntityPtr> const& refps, GeTransformCR transform)
{
	for (BPEntityPtr ptrRef : refps)
	{
		if (ptrRef.isNull())
			continue;

		BPProjectP  pProject = ptrRef->getBPProject();
		if (pProject == nullptr)
			continue;

		
		//根据传入的BPEntity信息获取对象实例
		BIMBase::Core::BPDataPtr ptrData = BPDataUtil::getDataOnEntity(*ptrRef);
		if (!ptrData.isValid())
			continue;

		ExampleTubeDemo pbSolid;
		pbSolid.initFromData(*ptrData);
		GePoint3d sPoint = pbSolid.getStartPoint();
		GePoint3d ePoint = pbSolid.getEndPoint();;

		transform.multiply(sPoint);
		transform.multiply(ePoint);

		pbSolid.setStartPoint(sPoint);
		pbSolid.setEndPoint(ePoint);
		pbSolid.onTransform(transform);

		P3DStatus statusAdd = pbSolid.replaceInProject(*pProject);
	}

}


class ToolExampleTubeRotateFactory :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		ToolExampleTubeRotate* p = new ToolExampleTubeRotate();
		p->AddRef();
		return p;
	}
};
static ToolExampleTubeRotateFactory s_ToolExampleTubeRotateFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("ExampleTubeDemo", IToolNameRotate, &s_ToolExampleTubeRotateFactory);
AutoDoRegisterFunctionsEnd