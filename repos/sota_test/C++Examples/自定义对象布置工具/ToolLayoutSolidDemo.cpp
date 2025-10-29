#include "stdafx.h"
#include "ToolLayoutSolidDemo.h"
#include "DlgSolidTypeDemo.h"
#include "ToolLayoutSolidDemo.h"

static DlgSolidTypeDemo* m_dlg;
ToolLayoutSolidDemo::ToolLayoutSolidDemo()
{
	m_eSolidType = GeSolidBaseType::GeSolidBaseType_TorusPipe;
	m_nRotStep = 0;
}


ToolLayoutSolidDemo::~ToolLayoutSolidDemo()
{
	m_vctPts.clear();
	if (m_dlg != nullptr)
		m_dlg->ShowWindow(SW_HIDE);
}

void ToolLayoutSolidDemo::_onPostInstall()
{
	//调用基类
	T_Super::_onPostInstall();
	PBBimModuleResourceOverride resOverride;
	if (m_dlg == nullptr)
	{
		m_dlg = new DlgSolidTypeDemo;
		m_dlg->Create(DlgSolidTypeDemo::IDD, AfxGetMainWnd());
		m_dlg->ShowWindow(SW_SHOW);
		m_eSolidType = m_dlg->m_eType;
		m_Solid.setSolidType(m_eSolidType);
	}
	else
		m_dlg->ShowWindow(SW_SHOW);
	//打开捕捉
	BPSnap::getInstance().enableLocate(false);
	BPSnap::getInstance().enableSnap(true);

	switch (m_eSolidType)
	{
	case p3d::GeSolidBaseType::GeSolidBaseType_None:
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_TorusPipe:
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择大圆圆心点"));
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_Cone:
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择底面圆心点"));
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_Box:
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_Sphere:
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择球心"));
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_Extrusion:
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"将布置一个圆环，请选择圆环底面圆心"));
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_RotationalSweep:
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"将布置一个壳体，请选择圆心点"));
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_RuledSweep:
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请在高度方向选择扫掠第一点"));
		break;
	default:
		break;
	}
}

void ToolLayoutSolidDemo::_onRestartTool()
{
	//重启工具
	ToolLayoutSolidDemo* newTool = new ToolLayoutSolidDemo();
	newTool->installTool();
}

bool ToolLayoutSolidDemo::_onDataButton(BPBaseButtonEventCP ev)
{
	//获取鼠标屏幕点击的点
	GePoint3d ptCur = *ev->getPoint();

	//获取点击点所在的工程和模型ID
	BPProjectP pProject = ev->getViewport()->getTargetModel()->getBPProject();
	::BIMBase::PModelId curModelId = ev->getViewport()->getTargetModel()->getModelId();


	//屏幕点击点Z向高度设置为楼层标高
	//ptCur.z = 0;
	m_vctPts.push_back(ptCur);
	int nSize = m_vctPts.size();
	switch (m_eSolidType)
	{
	case p3d::GeSolidBaseType::GeSolidBaseType_None:
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_TorusPipe:
		if (nSize == 1)
		{
			BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择圆管圆心点"));
		}
		if (nSize == 2)
		{
			BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择圆管半径"));
		}
		if (nSize == 3)
		{
			BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择扫掠终点"));
		}
		if (nSize == 4)
		{
			__addSolid(curModelId);
		}
			
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_Cone:
		if (nSize == 1)
		{
			BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择底面半径"));
		}
		if (nSize == 2)
		{
			BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择圆锥顶点"));
		}
		if (nSize == 3)
		{
			__addSolid(curModelId);
		}
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_Box:
		if (nSize == 1)
		{
			__addSolid(curModelId);
		}
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_Sphere:
		if (nSize == 1)
		{
			BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择球体半径"));
		}
		if (nSize == 2)
		{
			__addSolid(curModelId);
		}
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_Extrusion:
		if (nSize == 1)
		{
			BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择圆环内径点"));
		}
		if (nSize == 2)
		{
			BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择圆环外径点"));
		}
		if (nSize == 3)
		{
			BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择圆环高度"));
		}
		if (nSize == 4)
		{
			__addSolid(curModelId);
		}
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_RotationalSweep:
		if (nSize == 1)
		{
			//PBMessageCenter::Send(PBBim_MESSAGE_ToolPrompt, )
			BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择内径"));
		}
		if (nSize == 2)
		{
			BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择外径点"));
		}
		if (nSize == 3)
		{
			BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择扫掠终点"));
		}
		if (nSize == 4)
		{
			__addSolid(curModelId);
		}
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_RuledSweep:
		if (nSize == 1)
		{
			BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请在高度方向选择扫掠第二点"));
		}
		if (nSize == 2)
		{
			BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请在高度方向选择扫掠第三点"));
		}
		if (nSize == 3)
		{
			__addSolid(curModelId);
		}
		break;
	default:
		break;
	}
	

	//获取当前视图，强制刷新界面
	BPViewportP pViewPort = BPViewManager::getInstance().getActivedViewport();
	if (NULL == pViewPort) return false;
	if (pViewPort)
	{
		pViewPort->updateView();
	}

	return true;
}

bool ToolLayoutSolidDemo::_onResetButton(BPBaseButtonEventCP)
{
	m_vctPts.clear();
	_exitTool();
	return true;
}

void ToolLayoutSolidDemo::_onDynamicFrame(BPBaseButtonEventCP ev)
{
	if (NULL == ev)
		return;
	
	if (m_dlg != nullptr)
	{
		m_eSolidType = m_dlg->m_eType;
		m_Solid.setSolidType(m_eSolidType);
	}
		

	//获取鼠标屏幕点击的点
	GePoint3d ptDynamic = *ev->getPoint();

	//获取点击点所在的工程和模型ID
	BPProjectP pProject = ev->getViewport()->getTargetModel()->getBPProject();
	::BIMBase::PModelId curModelId = ev->getViewport()->getTargetModel()->getModelId();
	
	vector<GePoint3d> vctPoints(m_vctPts);
	vctPoints.push_back(ptDynamic);
	m_Solid.setPoints(vctPoints);
	//根据构造的数据随着鼠标移动动态显示构件
	BPGraphicsPtr ptrGraphics = m_Solid.createPhysicalGraphics(*pProject, curModelId, true);

	if (!ptrGraphics.isValid())
		return;

	BPGraphicsUtils::transformPhysicalGraphics(*ptrGraphics, m_Solid.getPlacement().toTransform());
	ptrGraphics->finish();

	BPRedrawEntitys redrawElems;
	if (m_eSolidType == p3d::GeSolidBaseType::GeSolidBaseType_Box)
	{
		BPPlacement placeOri = m_Solid.getPlacement();
		GeTransform trans = placeOri.toTransform();
		GeTransform transInv;
		transInv.setByInverse(trans);
		GeTransform transRot = GeTransform::createByAxisAndRotationAngle(GeRay3d::createByOriginAndVector(GePoint3d::create(0, 0, 0), GeVec3d::create(0, 0, 1)), m_nRotStep*PI / 2);
		GeTransform transRes = GeTransform::createByProduct(trans, GeTransform::createByProduct(transRot, transInv));
		redrawElems.setTransform(&transRes);
	}
	redrawElems.setDrawMode(BPDrawMode::enTempDraw);
	redrawElems.setDrawPurpose(BPDrawPurpose::enDynamics);
	redrawElems.setDynamicsViews(ev->getViewport());
	redrawElems.doRedraw(ptrGraphics->getEntityR());
}

bool ToolLayoutSolidDemo::_onModelMotion(BPBaseButtonEventCP ev)
{
	//如果动态没有开启则开启动态，这样才可以进入_OnDynamicFrame函数
	if (!getDynamicsStarted())
		_beginDynamics();
	return true;
}

bool ToolLayoutSolidDemo::_onKeyTransition(bool wentDown, ::p3d::platform::P3DVirtualKey key, bool shiftIsDown, bool ctrlIsDown)
{
	//设置box造型旋转
	switch (key)
	{
	case ::p3d::platform::P3DVirtualKey::Shift:
	{
		m_nRotStep = (m_nRotStep++) % 4;
	}
		break;
	}
	return true;
}

void ToolLayoutSolidDemo::__addSolid(PModelId modelId)
{
	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return;
	BPProjectP pProject = pProjectManager->getMainProject();
	if (pProject == NULL)
		return;

	m_Solid.setPoints(m_vctPts);

	
	//增加构件到工程中
	if (SUCCESS != m_Solid.addToProject(*pProject, modelId))
	{
		AfxMessageBox(L"Can not add to project!");
		m_vctPts.clear();
		return;
	}


	m_vctPts.clear();
	m_Solid.setPoints(m_vctPts);
}


BPPlacement ToolLayoutSolidDemo::cacuPlacement()
{
	BPPlacement placeOri = m_Solid.getPlacement();
	GeTransform trans = placeOri.toTransform();
	return placeOri;
}

//对工具进行注册
BPTool* CreateSolidDemoTool()
{
	ToolLayoutSolidDemo* tool = new ToolLayoutSolidDemo();
	return tool;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("layoutSolidDemo", &CreateSolidDemoTool);
AutoDoRegisterFunctionsEnd