#include "stdafx.h"
#include "ToolCubeMoveDemo.h"
#include "CubeDemo.h"

using namespace DemoObject;
ToolCubeMoveDemo::ToolCubeMoveDemo()
{
}


ToolCubeMoveDemo::~ToolCubeMoveDemo()
{
}

void ToolCubeMoveDemo::Dynamic(std::vector<BPEntityPtr> const & refps, GeTransformCR transform, BPRedrawEntitys& redrawElems)
{

}

void ToolCubeMoveDemo::Move(std::vector<BPEntityPtr> const & refps, GeTransformCR transform)
{
	for (BPEntityPtr ptrRef : refps)
	{
		if (ptrRef.isNull())
			continue;
		//根据传入的BPEntity信息获取对象实例
		BIMBase::Core::BPDataPtr ptrData = BPDataUtil::getDataOnEntity(*ptrRef);
		if (!ptrData.isValid())
			continue;
				
		CubeDemo pbCube;
		pbCube.initFromData(*ptrData);

		//选中的墙乘以移动的转换矩阵，移动到点击的位置
		pbCube.onTransform(transform);

		BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
		if (pProject == nullptr)
			return;

		//查找关联对象
		BPDataKeyArray keyArray;
		BPDataKey instanceKey = pbCube.getDataKey();
		BPRelationshipFinder::getTargetDatasBySource(keyArray, instanceKey, *pProject,PBM_SCHEMA_Demo, PBM_RELSHIP_CUBEWITHOPENNING);

		for (auto key : keyArray)
		{
			//实例化对象
			BPDataPtr ptrBallIns = BPDataUtil::getDataByKey(key, *ptrRef->getBPProject());
			if (!ptrBallIns.isValid())
				continue;

			OpenningDemo pbBall;
			pbBall.initFromData(*ptrBallIns);

			//关联对象乘以移动的转换矩阵，移动到点击的位置
			pbBall.onTransform(transform);

			//新位置的替换旧位置的
			pbBall.replaceInProject(*ptrRef->getBPProject());
		}
		//新位置的替换旧位置的
		pbCube.replaceInProject(*ptrRef->getBPProject());
		
	}
}

void ToolCubeMoveDemo::ElementsSelected(std::vector<BPEntityPtr> & refps)
{

}

//注册移动
class CubeMoveDemoFactory :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		ToolCubeMoveDemo *p = new ToolCubeMoveDemo();
		p->AddRef();
		return p;
	}
};
static CubeMoveDemoFactory s_CubeMoveDemoFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("CubeDemo", IToolNameMove, &s_CubeMoveDemoFactory);
AutoDoRegisterFunctionsEnd
