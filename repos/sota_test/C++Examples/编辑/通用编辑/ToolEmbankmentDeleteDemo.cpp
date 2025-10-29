#include "stdafx.h"
#include "ToolEmbankmentDeleteDemo.h"
#include "EmbankmentDemo.h"

using namespace DemoObject;

void ToolEmbankmentDeleteDemo::ElementsSelected(std::vector<::BIMBase::Core::BPEntityPtr>& refps)
{

}

TIErrorStatus ToolEmbankmentDeleteDemo::Delete(std::vector<::BIMBase::Core::BPEntityPtr> const& refps)
{
	for (BPEntityPtr ptrRef : refps)
	{
		if (ptrRef.isNull())
			continue;
		//根据传入的BPEntity信息获取对象实例
		BIMBase::Core::BPDataPtr ptrData = BPDataUtil::getDataOnEntity(*ptrRef);
		if (!ptrData.isValid())
			continue;

		EmbankmentDemo pbEmbankment;
		pbEmbankment.initFromData(*ptrData);

		pbEmbankment.deleteFromProject(*ptrRef->getBPProject());
	}

	return TIErrorStatus::succeed;
}

//注册移动
class EmbankmentDeleteDemoFactory :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		ToolEmbankmentDeleteDemo* p = new ToolEmbankmentDeleteDemo();
		p->AddRef();
		return p;
	}
};
static EmbankmentDeleteDemoFactory s_EmbankmentDeleteDemoFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("EmbankmentDemo", IToolNameDelete, &s_EmbankmentDeleteDemoFactory);
AutoDoRegisterFunctionsEnd