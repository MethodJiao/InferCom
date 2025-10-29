#include "stdafx.h"
#include "ToolDrainDeleteDemo.h"
#include "DrainDemo.h"

using namespace DemoObject;

void ToolDrainDeleteDemo::ElementsSelected(std::vector<::BIMBase::Core::BPEntityPtr>& refps)
{

}

TIErrorStatus ToolDrainDeleteDemo::Delete(std::vector<::BIMBase::Core::BPEntityPtr> const& refps)
{
	for (BPEntityPtr ptrRef : refps)
	{
		if (ptrRef.isNull())
			continue;
		//根据传入的BPEntity信息获取对象实例
		BIMBase::Core::BPDataPtr ptrData = BPDataUtil::getDataOnEntity(*ptrRef);
		if (!ptrData.isValid())
			continue;

		DrainDemo pbDrain;
		pbDrain.initFromData(*ptrData);

		pbDrain.deleteFromProject(*ptrRef->getBPProject());
	}

	return TIErrorStatus::succeed;
}

//注册移动
class DrainDeleteDemoFactory :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		ToolDrainDeleteDemo* p = new ToolDrainDeleteDemo();
		p->AddRef();
		return p;
	}
};
static DrainDeleteDemoFactory s_DrainDeleteDemoFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("DrainDemo", IToolNameDelete, &s_DrainDeleteDemoFactory);
AutoDoRegisterFunctionsEnd
