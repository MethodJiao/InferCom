#include "stdafx.h"


using namespace DemoObject;

void ToolExampleTubeCopy::ElementsSelected(std::vector<::BIMBase::Core::BPEntityPtr>& refps)
{

}

void ToolExampleTubeCopy::Dynamic(std::vector<::BIMBase::Core::BPEntityPtr> const& refps, p3d::GeTransformCR transform, ::BIMBase::Core::BPRedrawEntitys& redrawEntitys)
{

}

void ToolExampleTubeCopy::Copy(std::vector<::BIMBase::Core::BPEntityPtr> const& refps, p3d::GeTransformCR transform, unsigned int ncopy)
{

	for (BPEntityPtr ptrRef : refps)
	{
		if (ptrRef.isNull())
			continue;

		BPProjectP pProject = ptrRef->getBPProject();
		if (pProject == nullptr)
			continue;

		//根据传入的BPEntity信息获取对象实例
		BIMBase::Core::BPDataPtr ptr = BPDataUtil::getDataOnEntity(*ptrRef);
		if (!ptr.isValid())
			continue;


		ExampleTubeDemo pbHexagram;
		pbHexagram.initFromData(*ptr);

		//选中的六角星乘以移动的转换矩阵，移动到点击的位置
		pbHexagram.onTransform(transform);

		pbHexagram.addToProject(*pProject, pProject->getActiveModel()->getModelId());


	}


}


//注册复制
class ToolExampleTubeCopyFactory :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		ToolExampleTubeCopy* p = new ToolExampleTubeCopy();
		p->AddRef();
		return p;
	}
};
static ToolExampleTubeCopyFactory s_ToolExampleTubeCopyFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("ExampleTubeDemo", IToolNameCopy, &s_ToolExampleTubeCopyFactory);
AutoDoRegisterFunctionsEnd