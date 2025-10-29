#include "stdafx.h"
#include "ToolDrainMoveDemo.h"
#include "DrainDemo.h"

using namespace DemoObject;
ToolDrainMoveDemo::ToolDrainMoveDemo()
{
}


ToolDrainMoveDemo::~ToolDrainMoveDemo()
{
}

void ToolDrainMoveDemo::Dynamic(std::vector<BPEntityPtr> const& refps, GeTransformCR transform, BPRedrawEntitys& redrawElems)
{

}

void ToolDrainMoveDemo::Move(std::vector<BPEntityPtr> const& refps, GeTransformCR transform)
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

		//选中的墙乘以移动的转换矩阵，移动到点击的位置
		pbDrain.onTransform(transform);

		//新位置的替换旧位置的
		pbDrain.replaceInProject(*ptrRef->getBPProject());



	}
}

void ToolDrainMoveDemo::ElementsSelected(std::vector<BPEntityPtr>& refps)
{

}

//注册移动
class DrainMoveDemoFactory :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		ToolDrainMoveDemo* p = new ToolDrainMoveDemo();
		p->AddRef();
		return p;
	}
};
static DrainMoveDemoFactory s_DrainMoveDemoFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("DrainDemo", IToolNameMove, &s_DrainMoveDemoFactory);
AutoDoRegisterFunctionsEnd
