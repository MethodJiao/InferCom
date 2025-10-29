#include "stdafx.h"



using namespace DemoObject;
ToolExampleTubeMove::ToolExampleTubeMove()
{
}


ToolExampleTubeMove::~ToolExampleTubeMove()
{
}

void ToolExampleTubeMove::Dynamic(std::vector<BPEntityPtr> const& refps, GeTransformCR transform, BPRedrawEntitys& redrawElems)
{

}

void ToolExampleTubeMove::Move(std::vector<BPEntityPtr> const& refps, GeTransformCR transform)
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
		pbCube.setStartPoint(GePoint3d::createByTransform(transform, pbCube.getStartPoint()));
		pbCube.setEndPoint(GePoint3d::createByTransform(transform, pbCube.getEndPoint()));
		
		//选中的墙乘以移动的转换矩阵，移动到点击的位置
		pbCube.onTransform(transform);

		//新位置的替换旧位置的
		::p3d::P3DStatus sta =pbCube.replaceInProject(*ptrRef->getBPProject());
		int a = 1;

	}
}

void ToolExampleTubeMove::ElementsSelected(std::vector<BPEntityPtr>& refps)
{

}

//注册移动
class ExampleTubeMoveDemoFactory :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		ToolExampleTubeMove* p = new ToolExampleTubeMove();
		p->AddRef();
		return p;
	}
};
static ExampleTubeMoveDemoFactory s_ExampleTubeMoveDemoFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("ExampleTubeDemo", IToolNameMove, &s_ExampleTubeMoveDemoFactory);
AutoDoRegisterFunctionsEnd
