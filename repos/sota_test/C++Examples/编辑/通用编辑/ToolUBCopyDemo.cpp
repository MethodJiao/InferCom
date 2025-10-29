#include "stdafx.h"
#include "ToolUBCopyDemo.h"
#include "UniversalBeamDemo.h"

using namespace DemoObject;
void ToolUBCopyDemo::ElementsSelected(std::vector<::BIMBase::Core::BPEntityPtr>& refps)
{

}

void ToolUBCopyDemo::Dynamic(std::vector<::BIMBase::Core::BPEntityPtr> const& refps, p3d::GeTransformCR transform, ::BIMBase::Core::BPRedrawEntitys& redrawEntitys)
{

}

void ToolUBCopyDemo::Copy(std::vector<::BIMBase::Core::BPEntityPtr> const& refps, p3d::GeTransformCR transform, unsigned int ncopy)
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

		IBPObjectPtr ptrObjCopy = BPObjectExtensionManager::getInstance().getBPObject(*pProject, ptrData->getDataKey());
		if (ptrObjCopy.isNull())
			continue;

		UniversalBeamDemoPtr ptrUBCopy = dynamic_cast<UniversalBeamDemoP>(ptrObjCopy.get());
		if (ptrUBCopy.isNull())
			continue;

		ptrUBCopy->onTransform(transform);

		ptrUBCopy->addToProject(*pProject, pProject->getActiveModel()->getModelId());
	}
}

//注册复制
class UBCopyDemoFactory :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		ToolUBCopyDemo* p = new ToolUBCopyDemo();
		p->AddRef();
		return p;
	}
};
static UBCopyDemoFactory s_UBCopyDemoFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("UniversalBeamDemo", IToolNameCopy, &s_UBCopyDemoFactory);
AutoDoRegisterFunctionsEnd