#include "stdafx.h"
#include "InsulatorDeleteDemo.h"
#include "InsulatorDemo.h"


using namespace DemoObject;
using namespace BIMBase;
using namespace BIMBase::Data;

void InsulatorDeleteDemo::ElementsSelected(std::vector<::BIMBase::Core::BPEntityPtr>& refps)
{

}

TIErrorStatus InsulatorDeleteDemo::Delete(std::vector<::BIMBase::Core::BPEntityPtr> const& refps)
{
	for (BPEntityPtr ptrRef : refps)
	{
		if (ptrRef.isNull())
			continue;

		BPProjectP pProject = ptrRef->getBPProject();
		if (pProject == nullptr)
			continue;

		//根据传入的BPEntity信息获取对象实例
		BIMBase::Core::BPDataPtr ptrData = BPDataUtil::getDataOnEntity(*ptrRef);
		if (!ptrData.isValid())
			continue;

		IBPObjectPtr ptrObj = BPObjectExtensionManager::getInstance().getBPObject(*pProject, ptrData->getDataKey());
		if (ptrObj.isNull())
			continue;

		InsulatorDemoPtr ptrInsulator = dynamic_cast<InsulatorDemo*>(ptrObj.get());
		if (ptrInsulator.isNull())
			continue;

		ptrInsulator->deleteFromProject(*ptrRef->getBPProject());

	}
	return TIErrorStatus::succeed;
}
//注册删除
class BPInsulatorDemoFactory :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		InsulatorDeleteDemo* p = new InsulatorDeleteDemo();
		p->AddRef();
		return p;
	}
};
static BPInsulatorDemoFactory s_InsulatorDeleteDemoFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("InsulatorDemo", IToolNameDelete, &s_InsulatorDeleteDemoFactory);
AutoDoRegisterFunctionsEnd