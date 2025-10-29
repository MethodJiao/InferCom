#include "stdafx.h"
#include "DlgArchWallDemo.h"
#include "ToolArchWallDemo.h"

static DlgArchWallDemo* m_dlg;
ToolArchWallDemo::ToolArchWallDemo()
{
	m_eLayoutWay = ToolLayoutCubeDemo::CubeLayoutWay::OnePoint;
	m_nHeight = 3000;
	m_nLength = 1000;
	m_nWidth = 200;
}


ToolArchWallDemo::~ToolArchWallDemo()
{
	if (m_dlg != nullptr)
		m_dlg->ShowWindow(SW_HIDE);
}

void ToolArchWallDemo::_onPostInstall()
{
	//调用基类
	T_Super::_onPostInstall();

	PBBimModuleResourceOverride resOverride;
	if (m_dlg == nullptr)
	{
		CView* pView = ((CFrameWnd*)(AfxGetApp()->m_pMainWnd))->GetActiveFrame()->GetActiveView();
		m_dlg = new DlgArchWallDemo;
		m_dlg->Create(DlgArchWallDemo::IDD, pView);
		m_dlg->ShowWindow(SW_SHOW);
		m_eLayoutWay = (ToolLayoutCubeDemo::CubeLayoutWay)m_dlg->m_nInputWay;
	}
	else
		m_dlg->ShowWindow(SW_SHOW);
	//打开捕捉
	BPSnap::getInstance().enableLocate(false);
	BPSnap::getInstance().enableSnap(true);
}

void ToolArchWallDemo::_onRestartTool()
{
	//重启工具
	ToolArchWallDemo* newTool = new ToolArchWallDemo();
	newTool->installTool();
}

bool ToolArchWallDemo::_onDataButton(BPBaseButtonEventCP ev)
{
	//获取鼠标屏幕点击的点
	GePoint3d ptCur = *ev->getPoint();

	//获取点击点所在的工程和模型ID
	::BIMBase::PModelId curModelId = ev->getViewport()->getTargetModel()->getModelId();

	//屏幕点击点Z向高度设置为楼层标高
	ptCur.z = 0;
	int nSize = 0;
	switch (m_eLayoutWay)
	{
	case ToolLayoutCubeDemo::OnePoint:
		__createOnePtData(ptCur);
		__addCube(curModelId);
		break;
	case ToolLayoutCubeDemo::Draw:
		m_vctPts.push_back(ptCur);
		if (m_vctPts.size() != 2)
			return false;

		__createDrawData(m_vctPts[0], m_vctPts[1]);
		__addCube(curModelId);
		break;
	case ToolLayoutCubeDemo::Multi:
		m_vctPts.push_back(ptCur);
		nSize = m_vctPts.size();
		if (nSize < 2)
			return false;
		__createDrawData(m_vctPts[nSize - 2], m_vctPts[nSize - 1]);
		__addCube(curModelId);
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

bool ToolArchWallDemo::_onResetButton(BPBaseButtonEventCP)
{
	//点击右键退出工具
	_exitTool();
	return true;
}

void ToolArchWallDemo::_onDynamicFrame(BPBaseButtonEventCP ev)
{
	if (NULL == ev)
		return;

	if (m_dlg != nullptr)
		m_eLayoutWay = (ToolLayoutCubeDemo::CubeLayoutWay)m_dlg->m_nInputWay;

	//获取鼠标屏幕点击的点
	GePoint3d ptDynamic = *ev->getPoint();

	//获取点击点所在的工程和模型ID
	BPProjectP pProject = ev->getViewport()->getTargetModel()->getBPProject();
	::BIMBase::PModelId curModelId = ev->getViewport()->getTargetModel()->getModelId();

	//屏幕点击点Z向高度设置为楼层标高
	ptDynamic.z = 0;
	int nSize = 0;
	switch (m_eLayoutWay)
	{
	case ToolLayoutCubeDemo::OnePoint:
		__createOnePtData(ptDynamic);
		break;
	case ToolLayoutCubeDemo::Draw:
		if (m_vctPts.size() != 1)
			return;
		__createDrawData(m_vctPts[0], ptDynamic);
		break;
	case ToolLayoutCubeDemo::Multi:
		nSize = m_vctPts.size();
		if (nSize < 1)
			return;
		__createDrawData(m_vctPts[nSize - 1], ptDynamic);
		break;
	default:
		break;
	}


	//根据构造的墙数据随着鼠标移动动态显示墙构件
	BPGraphicsPtr ptrGraphics;
	ptrGraphics = m_Cube.createPhysicalGraphics(*pProject, curModelId, true);

	if (!ptrGraphics.isValid())
		return;

	BPGraphicsUtils::transformPhysicalGraphics(*ptrGraphics, m_Cube.getPlacement().toTransform());
	ptrGraphics->finish();

	BPRedrawEntitys redrawElems;
	redrawElems.setDrawMode(BPDrawMode::enTempDraw);
	redrawElems.setDrawPurpose(BPDrawPurpose::enDynamics);
	redrawElems.setDynamicsViews(ev->getViewport());
	redrawElems.doRedraw(ptrGraphics->getEntityR());
}


bool ToolArchWallDemo::_onModelMotion(BPBaseButtonEventCP ev)
{
	//如果动态没有开启则开启动态，这样才可以进入_OnDynamicFrame函数
	if (!getDynamicsStarted())
		_beginDynamics();
	return true;
}

bool ToolArchWallDemo::_onKeyTransition(bool wentDown, ::p3d::platform::P3DVirtualKey key, bool shiftIsDown, bool ctrlIsDown)
{
	//设置 左右键：单点布置   上下键：两点绘制
	switch (key)
	{
	case ::p3d::platform::P3DVirtualKey::Left:
	case ::p3d::platform::P3DVirtualKey::Right:
		m_eLayoutWay = ToolLayoutCubeDemo::CubeLayoutWay::OnePoint;
		break;
	case ::p3d::platform::P3DVirtualKey::Up:
	case ::p3d::platform::P3DVirtualKey::Down:
		m_eLayoutWay = ToolLayoutCubeDemo::CubeLayoutWay::Draw;
		break;
	}

	if (m_dlg != nullptr)
	{
		m_dlg->m_nInputWay = (int)m_eLayoutWay;
		m_dlg->updateUI();
	}
	return true;
}

void ToolArchWallDemo::__addCube(PModelId modelId)
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;

	//增加扩展属性
	BPExtendPropertySet extendPropertySet = m_Cube.getExtendPropertySet();
	BPPropertyValue valueExtend;
	valueExtend.m_value.m_valueString = _T("墙自定义属性文本");
	valueExtend.m_type = BPPropertyValue::String;
	extendPropertySet.setSubParam(_T("墙自定义属性"), valueExtend);
	m_Cube.setExtendPropertySet(extendPropertySet);

	//增加构件到工程中
	if (SUCCESS != m_Cube.addToProject(*pProject, modelId))
	{
		AfxMessageBox(L"无法添加到工程，DemoSchema.01.00.pkpmschema.dat是否正确!");
	}

	BPDataKey key = m_Cube.getDataKey();
	BPEntityId entityId = BPEntityUtil::getPrimaryEntityWithData(*pProject, key, modelId);
	BPEntity entity(entityId, *pProject);

	//添加到工程后修改颜色--------------------------------------------------------

	BPGraphicsPtr ptrGraphic = BPGraphics::getGraphicsFromEntity(entity);
	if (ptrGraphic.isNull())
		return;

	BIMBase::BPColorDef colorDef(141, 141, 141);

	UInt32 nColor = BPColorUtil::getEntityColor(colorDef, *pProject, true);
	UInt32 nWeight = 0, nColor2 = 0; Int32 nStyle = 0;

	BPSymbology sys;
	sys.color = nColor;
	sys.weight = nWeight;
	sys.style = nStyle;
	ptrGraphic->setSymbologySource(BPSymbologySource::enEntity);
	ptrGraphic->setSymbology(sys);
	ptrGraphic->updateEntityWithGraphics(&entity);

	if (m_eLayoutWay != ToolLayoutCubeDemo::Multi)
		m_vctPts.clear();
}

void ToolArchWallDemo::__createOnePtData(GePoint3d ptOri)
{
	BPPlacement placementNew = m_Cube.getPlacement();
	placementNew.setOrigin(ptOri);

	//设置基本信息
	m_Cube.setPlacement(placementNew);
	m_Cube.setHeight(m_nHeight);
	m_Cube.setWidth(m_nWidth);
	m_Cube.setLength(m_nLength);
}

void ToolArchWallDemo::__createDrawData(GePoint3d ptOri, GePoint3d ptSecond)
{
	//根据长度方向确定X轴
	GeVec3d vecX = GeVec3d::createByStartEndNormalize(ptOri, ptSecond);
	GeVec3d vecZ = GeVec3d::create(0, 0, 1);
	//根据叉乘获取Y轴
	GeVec3d vecY = vecZ ^ vecX;
	m_nLength = ptOri.distance(ptSecond);

	//根据原点与xyz轴设置转换矩阵
	GeTransform tran = GeTransform::createByOriginAndVectors(ptOri, vecX, vecY, vecZ);
	BPPlacement placement;
	placement.fromTransform(tran);

	//设置墙基本信息
	m_Cube.setPlacement(placement);
	m_Cube.setHeight(m_nHeight);
	m_Cube.setWidth(m_nWidth);
	m_Cube.setLength(m_nLength);
}

//对工具进行注册
BPTool* CreateArchWallTool()
{
	ToolArchWallDemo* tool = new ToolArchWallDemo();
	return tool;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("ArchWallDemo", &CreateArchWallTool);
AutoDoRegisterFunctionsEnd