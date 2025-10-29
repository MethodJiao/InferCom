#include "stdafx.h"


using namespace DemoObject;

void ToolExampleTubeDelete::ElementsSelected(std::vector<::BIMBase::Core::BPEntityPtr>& refps)
{

}

TIErrorStatus ToolExampleTubeDelete::Delete(std::vector<::BIMBase::Core::BPEntityPtr> const& refps)
{
	for (BPEntityPtr ptrRef : refps)
	{
		if (ptrRef.isNull())
			continue;
		//根据传入的BPEntity信息获取对象实例
		BIMBase::Core::BPDataPtr ptrData = BPDataUtil::getDataOnEntity(*ptrRef);
		if (!ptrData.isValid())
			continue;

		ExampleTubeDemo pbCube;
		pbCube.initFromData(*ptrData);

		pbCube.deleteFromProject(*ptrRef->getBPProject());
	}

	return TIErrorStatus::succeed;
}

//注册删除
class ToolExampleTubeDeleteDemoFactory :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		ToolExampleTubeDelete* p = new ToolExampleTubeDelete();
		p->AddRef();
		return p;
	}
};
static ToolExampleTubeDeleteDemoFactory s_ExampleCubeDeleteDemoFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("ExampleTubeDemo", IToolNameDelete, &s_ExampleCubeDeleteDemoFactory);
AutoDoRegisterFunctionsEnd
