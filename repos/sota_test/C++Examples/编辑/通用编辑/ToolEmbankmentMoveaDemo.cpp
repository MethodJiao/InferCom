#include "stdafx.h"
#include "ToolEmbankmentMoveDemo.h"
#include "EmbankmentDemo.h"

using namespace DemoObject;
ToolEmbankmentMoveDemo::ToolEmbankmentMoveDemo()
{
}


ToolEmbankmentMoveDemo::~ToolEmbankmentMoveDemo()
{
}

void ToolEmbankmentMoveDemo::Dynamic(std::vector<BPEntityPtr> const& refps, GeTransformCR transform, BPRedrawEntitys& redrawElems)
{

}

void ToolEmbankmentMoveDemo::Move(std::vector<BPEntityPtr> const& refps, GeTransformCR transform)
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

		//选中的墙乘以移动的转换矩阵，移动到点击的位置
		pbEmbankment.onTransform(transform);

		//新位置的替换旧位置的
		pbEmbankment.replaceInProject(*ptrRef->getBPProject());



	}
}

void ToolEmbankmentMoveDemo::ElementsSelected(std::vector<BPEntityPtr>& refps)
{

}

//注册移动
class EmbankmentMoveDemoFactory :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		ToolEmbankmentMoveDemo* p = new ToolEmbankmentMoveDemo();
		p->AddRef();
		return p;
	}
};
static EmbankmentMoveDemoFactory s_EmbankmentMoveDemoFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("EmbankmentDemo", IToolNameMove, &s_EmbankmentMoveDemoFactory);
AutoDoRegisterFunctionsEnd
