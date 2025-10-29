#include "stdafx.h"
#include "ToolUBDeletDemo.h"
#include "UniversalBeamDemo.h"

using namespace DemoObject;

void ToolUBDeleteDemo::ElementsSelected(std::vector<::BIMBase::Core::BPEntityPtr>& refps)
{

}

TIErrorStatus ToolUBDeleteDemo::Delete(std::vector<::BIMBase::Core::BPEntityPtr> const& refps)
{
	for (BPEntityPtr ptrRef : refps)
	{
		if (ptrRef.isNull())
			continue;
		//根据传入的BPEntity信息获取对象实例
		BIMBase::Core::BPDataPtr ptrData = BPDataUtil::getDataOnEntity(*ptrRef);
		if (!ptrData.isValid())
			continue;

		UniversalBeamDemo pbUB;
		pbUB.initFromData(*ptrData);

		pbUB.deleteFromProject(*ptrRef->getBPProject());
	}

	return TIErrorStatus::succeed;
}

//注册移动
class UBDeleteDemoFactory :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		ToolUBDeleteDemo* p = new ToolUBDeleteDemo();
		p->AddRef();
		return p;
	}
};
static UBDeleteDemoFactory s_UBDeleteDemoFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory(PBM_CLASS_UNIVERSAL_BEAM_Demo, IToolNameDelete, &s_UBDeleteDemoFactory);
AutoDoRegisterFunctionsEnd
