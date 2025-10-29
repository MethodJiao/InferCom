#include "stdafx.h"
#include "ToolUBMoveDemo.h"
#include "UniversalBeamDemo.h"

using namespace DemoObject;
ToolUBMoveDemo::ToolUBMoveDemo()
{
}


ToolUBMoveDemo::~ToolUBMoveDemo()
{
}

void ToolUBMoveDemo::Dynamic(std::vector<BPEntityPtr> const& refps, GeTransformCR transform, BPRedrawEntitys& redrawElems)
{

}

void ToolUBMoveDemo::Move(std::vector<BPEntityPtr> const& refps, GeTransformCR transform)
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

		//选中的墙乘以移动的转换矩阵，移动到点击的位置
		pbUB.onTransform(transform);

		//新位置的替换旧位置的
		pbUB.replaceInProject(*ptrRef->getBPProject());
	}
}

void ToolUBMoveDemo::ElementsSelected(std::vector<BPEntityPtr>& refps)
{

}

//注册移动
class UBMoveDemoFactory :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		ToolUBMoveDemo* p = new ToolUBMoveDemo();
		p->AddRef();
		return p;
	}
};
static UBMoveDemoFactory s_UBMoveDemoFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory(PBM_CLASS_UNIVERSAL_BEAM_Demo, IToolNameMove, &s_UBMoveDemoFactory);
AutoDoRegisterFunctionsEnd
