#include "stdafx.h"
#include "ToolLayoutInsulatorDemo.h"

ToolLayoutInsulatorDemo::ToolLayoutInsulatorDemo()
{
	/**联数*/
	m_nN = 2;
	/**单串绝缘子片数量*/
	m_nN1 = 15;
	/**绝缘子单片连接高度*/
	m_dH1 = 100;
	/**大伞裙半径*/
	m_dR1 = 50;
	/**小伞裙半径*/
	m_dR2 = 40;
	/**绝缘子串半径*/
	m_dR = 30;
	/**双串间距*/
	m_dD = 110;
	/**前端长度（构架端*/
	m_dFL = 100;
	/**后端长度（导线端）*/
	m_dAL = 50;

}


ToolLayoutInsulatorDemo::~ToolLayoutInsulatorDemo()
{

}

void ToolLayoutInsulatorDemo::_onPostInstall()
{
	//调用基类
	T_Super::_onPostInstall();

	//打开捕捉
	BPSnap::getInstance().enableLocate(false);
	BPSnap::getInstance().enableSnap(true);
}

void ToolLayoutInsulatorDemo::_onRestartTool()
{
	//重启工具
	ToolLayoutInsulatorDemo* newTool = new ToolLayoutInsulatorDemo();
	newTool->installTool();
}

bool ToolLayoutInsulatorDemo::_onDataButton(BPBaseButtonEventCP ev)
{
	//获取鼠标屏幕点击的点
	GePoint3d ptCur = *ev->getPoint();

	//获取点击点所在的工程和模型ID
	::BIMBase::PModelId curModelId = ev->getViewport()->getTargetModel()->getModelId();

	//屏幕点击点Z向高度设置为楼层标高
	//ptCur.z = 0;
	//int nSize = 0;
	__createOnePtData(ptCur);
	__addInsulator(curModelId);


	//获取当前视图，强制刷新界面
	BPViewportP pViewPort = BPViewManager::getInstance().getActivedViewport();
	if (NULL == pViewPort) return false;
	if (pViewPort)
	{
		pViewPort->updateView();
	}

	return true;
}

bool ToolLayoutInsulatorDemo::_onResetButton(BPBaseButtonEventCP)
{
	//点击右键退出工具
	_exitTool();
	return true;
}

void ToolLayoutInsulatorDemo::_onDynamicFrame(BPBaseButtonEventCP ev)
{
	if (NULL == ev)
		return;


	//获取鼠标屏幕点击的点
	GePoint3d ptDynamic = *ev->getPoint();

	//获取点击点所在的工程和模型ID
	BPProjectP pProject = ev->getViewport()->getTargetModel()->getBPProject();
	::BIMBase::PModelId curModelId = ev->getViewport()->getTargetModel()->getModelId();

	//屏幕点击点Z向高度设置为楼层标高
	//ptDynamic.z = 0;
	//int nSize = 0;
	__createOnePtData(ptDynamic);


	//根据构造的墙数据随着鼠标移动动态显示构件
	BPGraphicsPtr ptrGraphics;
	ptrGraphics = m_Insulator.createPhysicalGraphics(*pProject, curModelId, true);

	if (!ptrGraphics.isValid())
		return;

	BPGraphicsUtils::transformPhysicalGraphics(*ptrGraphics, m_Insulator.getPlacement().toTransform());
	ptrGraphics->finish();

	BPRedrawEntitys redrawElems;
	redrawElems.setDrawMode(BPDrawMode::enTempDraw);
	redrawElems.setDrawPurpose(BPDrawPurpose::enDynamics);
	redrawElems.setDynamicsViews(ev->getViewport());
	redrawElems.doRedraw(ptrGraphics->getEntityR());
}


bool ToolLayoutInsulatorDemo::_onModelMotion(BPBaseButtonEventCP ev)
{
	//如果动态没有开启则开启动态，这样才可以进入_OnDynamicFrame函数
	if (!getDynamicsStarted())
		_beginDynamics();
	return true;
}



void ToolLayoutInsulatorDemo::__addInsulator(PModelId modelId)
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;

	//增加构件到工程中
	if (SUCCESS != m_Insulator.addToProject(*pProject, modelId))
	{
		AfxMessageBox(L"Can not add to project!");
	}

}

void ToolLayoutInsulatorDemo::__createOnePtData(GePoint3d ptOri)
{
	BPPlacement placementNew = m_Insulator.getPlacement();
	placementNew.setOrigin(ptOri);
	m_Insulator.setCenter(ptOri);
	m_Insulator.setN(m_nN);
	m_Insulator.setD(m_dD);
	m_Insulator.setN1(m_nN1);
	m_Insulator.setH1(m_dH1);
	m_Insulator.setR1(m_dR1);
	m_Insulator.setR2(m_dR2);
	m_Insulator.setR(m_dR);
	m_Insulator.setFL(m_dFL);
	m_Insulator.setAL(m_dAL);
}



//对工具进行注册
BPTool* CreateInsulatorDemoTool()
{
	ToolLayoutInsulatorDemo* tool = new ToolLayoutInsulatorDemo();
	return tool;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("layoutInsulatorDemo", &CreateInsulatorDemoTool);
AutoDoRegisterFunctionsEnd