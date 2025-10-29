#include "stdafx.h"
#include "ToolLayoutDrainDemo.h"

ToolLayoutDrainDemo::ToolLayoutDrainDemo()
{
	m_nWidth = 600;
	m_nThickness = 100;
	m_nDepth = 500;
	m_nLenght = 30000;
}


ToolLayoutDrainDemo::~ToolLayoutDrainDemo()
{
}

void ToolLayoutDrainDemo::_onPostInstall()
{
	//调用基类
	T_Super::_onPostInstall();

	//打开捕捉
	BPSnap::getInstance().enableLocate(false);
	BPSnap::getInstance().enableSnap(true);
}

void ToolLayoutDrainDemo::_onRestartTool()
{
	//重启工具
	ToolLayoutDrainDemo* newTool = new ToolLayoutDrainDemo();
	newTool->installTool();
}

bool ToolLayoutDrainDemo::_onDataButton(BPBaseButtonEventCP ev)
{
	//获取鼠标屏幕点击的点
	GePoint3d ptCur = *ev->getPoint();

	//获取点击点所在的工程和模型ID
	::BIMBase::PModelId curModelId = ev->getViewport()->getTargetModel()->getModelId();

	//屏幕点击点Z向高度设置为楼层标高
	ptCur.z = 0;

	__createOnePtData(ptCur);
	__addEmbankment(curModelId);

	//获取当前视图，强制刷新界面
	BPViewportP pViewPort = BPViewManager::getInstance().getActivedViewport();
	if (NULL == pViewPort) return false;
	if (pViewPort)
	{
		pViewPort->updateView();
	}

	return true;
}

bool ToolLayoutDrainDemo::_onResetButton(BPBaseButtonEventCP)
{
	//点击右键退出工具
	_exitTool();
	return true;
}

void ToolLayoutDrainDemo::_onDynamicFrame(BPBaseButtonEventCP ev)
{
	if (NULL == ev)
		return;

	//获取鼠标屏幕点击的点
	GePoint3d ptDynamic = *ev->getPoint();

	//获取点击点所在的工程和模型ID
	BPProjectP pProject = ev->getViewport()->getTargetModel()->getBPProject();
	::BIMBase::PModelId curModelId = ev->getViewport()->getTargetModel()->getModelId();

	//屏幕点击点Z向高度设置为楼层标高
	ptDynamic.z = 0;
	__createOnePtData(ptDynamic);
	//根据构造的墙数据随着鼠标移动动态显示墙构件
	BPGraphicsPtr ptrGraphics;
	ptrGraphics = m_Embankment.createPhysicalGraphics(*pProject, curModelId, true);

	if (!ptrGraphics.isValid())
		return;

	BPGraphicsUtils::transformPhysicalGraphics(*ptrGraphics, m_Embankment.getPlacement().toTransform());
	ptrGraphics->finish();

	BPRedrawEntitys redrawElems;
	redrawElems.setDrawMode(BPDrawMode::enTempDraw);
	redrawElems.setDrawPurpose(BPDrawPurpose::enDynamics);
	redrawElems.setDynamicsViews(ev->getViewport());
	redrawElems.doRedraw(ptrGraphics->getEntityR());
}


bool ToolLayoutDrainDemo::_onModelMotion(BPBaseButtonEventCP ev)
{
	//如果动态没有开启则开启动态，这样才可以进入_OnDynamicFrame函数
	if (!getDynamicsStarted())
		_beginDynamics();
	return true;
}


void ToolLayoutDrainDemo::__addEmbankment(PModelId modelId)
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;


	//增加构件到工程中
	if (SUCCESS != m_Embankment.addToProject(*pProject, modelId))
	{
		AfxMessageBox(L"Can not add to project!");
	}


}

void ToolLayoutDrainDemo::__createOnePtData(GePoint3d ptOri)
{
	BPPlacement placementNew = m_Embankment.getPlacement();
	placementNew.setOrigin(ptOri);

	//设置基本信息
	m_Embankment.setPlacement(placementNew);
	m_Embankment.setWidth(m_nWidth);
	m_Embankment.setLength(m_nLenght);
	m_Embankment.setThickness(m_nThickness);
	m_Embankment.setDepth(m_nDepth);
}



//对工具进行注册
BPTool* CreateDrainDemoTool()
{
	ToolLayoutDrainDemo* tool = new ToolLayoutDrainDemo();
	return tool;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("layoutDrainDemo", &CreateDrainDemoTool);
AutoDoRegisterFunctionsEnd