#include "stdafx.h"
#include "ToolCubeCopyDemo.h"
#include "CubeDemo.h"

using namespace DemoObject;
void ToolCubeCopyDemo::ElementsSelected(std::vector<::BIMBase::Core::BPEntityPtr>& refps)
{

}

void ToolCubeCopyDemo::Dynamic(std::vector<::BIMBase::Core::BPEntityPtr> const& refps, p3d::GeTransformCR transform, ::BIMBase::Core::BPRedrawEntitys& redrawEntitys)
{

}

void ToolCubeCopyDemo::Copy(std::vector<::BIMBase::Core::BPEntityPtr> const& refps, p3d::GeTransformCR transform, unsigned int ncopy)
{
	for (BPEntityPtr ptrRef : refps)
	{
		if (ptrRef.isNull())
			continue;

		BPProjectP pProject = ptrRef->getBPProject();
		if(pProject == nullptr)
			continue;

		//根据传入的BPEntity信息获取对象实例
		BIMBase::Core::BPDataPtr ptrData = BPDataUtil::getDataOnEntity(*ptrRef);
		if (!ptrData.isValid())
			continue;

		IBPObjectPtr ptrObjCopy = BPObjectExtensionManager::getInstance().getBPObject(*pProject, ptrData->getDataKey());
		if(ptrObjCopy.isNull())
			continue;

		BPGraphicElementPtr ptrCubeCopy = dynamic_cast<BPGraphicElement*>(ptrObjCopy.get());
		if(ptrCubeCopy.isNull())
			continue;

		ptrCubeCopy->onTransform(transform);

		ptrCubeCopy->addToProject(*pProject, pProject->getActiveModel()->getModelId());
	}
}

//注册复制
class CubeCopyDemoFactory :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		ToolCubeCopyDemo* p = new ToolCubeCopyDemo();
		p->AddRef();
		return p;
	}
};
static CubeCopyDemoFactory s_CubeCopyDemoFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("CubeDemo", IToolNameCopy, &s_CubeCopyDemoFactory);
AutoDoRegisterFunctionsEnd