#include "stdafx.h"
#include "ToolCubeDeleteDemo.h"
#include "CubeDemo.h"

using namespace DemoObject;

void ToolCubeDeleteDemo::ElementsSelected(std::vector<::BIMBase::Core::BPEntityPtr>& refps)
{

}

TIErrorStatus ToolCubeDeleteDemo::Delete(std::vector<::BIMBase::Core::BPEntityPtr> const& refps)
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

		IBPObjectPtr ptrObjDelete = BPObjectExtensionManager::getInstance().getBPObject(*pProject, ptrData->getDataKey());
		if (ptrObjDelete.isNull())
			continue;

		BPGraphicElementPtr ptrEleDelete = dynamic_cast<BPGraphicElement*>(ptrObjDelete.get());
		if (ptrEleDelete.isNull())
			continue;

		ptrEleDelete->deleteFromProject(*ptrRef->getBPProject());
	}

	return TIErrorStatus::succeed;
}

//注册移动
class CubeDeleteDemoFactory :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		ToolCubeDeleteDemo* p = new ToolCubeDeleteDemo();
		p->AddRef();
		return p;
	}
};
static CubeDeleteDemoFactory s_CubeDeleteDemoFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("CubeDemo", IToolNameDelete, &s_CubeDeleteDemoFactory);
AutoDoRegisterFunctionsEnd
