#include "stdafx.h"
#include "ToolLayoutArcDemo.h"

ToolLayoutArcDemo::ToolLayoutArcDemo()
{
	m_nStep = 0;
	m_nType = 0;
}

ToolLayoutArcDemo::~ToolLayoutArcDemo()
{}

void ToolLayoutArcDemo::_onPostInstall()
{
	T_Super::_onPostInstall();
	m_nType = 0;
	BPSnap::getInstance().enableLocate(false);
	BPSnap::getInstance().enableSnap(true);
	showInputDlg(false);//关闭平台追踪器
}

void ToolLayoutArcDemo::_onRestartTool()
{
	ToolLayoutArcDemo* newTool = new ToolLayoutArcDemo();
	newTool->installTool();
}

bool ToolLayoutArcDemo::_onDataButton(BPBaseButtonEventCP ev)
{
	if (m_nType == 0)
	{
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"当前是角度标注，按shift键可以切换为弧长标注"));
	}
	if (m_nType == 1)
	{
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"当前是弧长标注，按shift键可以切换为角度标注"));
	}		
	if (m_nStep == 0)//点击第一个点
	{
		m_ptF = *ev->getPoint();		
		m_nStep = 1;
	}
	else if (m_nStep == 1)//点击第二个点
	{
		m_ptS = *ev->getPoint();
		m_nStep = 2;
	}
	else
	{
		if (m_ptrGraphic.isNull())
			return false;
		m_ptrGraphic->save();
		m_nStep = 0;
	}
	return true;
}

bool ToolLayoutArcDemo::_onResetButton(BPBaseButtonEventCP)
{
	_exitTool();
	return true;
}

void ToolLayoutArcDemo::_onDynamicFrame(BPBaseButtonEventCP ev)
{
	if (NULL == ev)
		return;
	if (m_nStep == 0 || m_nStep == 1)
		return;

	BPModelP pModel = ev->getViewport()->getTargetModel();
	if (pModel == nullptr)
		return;
	BIMBase::Core::BPProjectP pProject = pModel->getBPProject();
	if (pProject == nullptr)
		return;

	GePoint3d ptE = *ev->getPoint();
	__createGraphic(ptE, pProject, pModel);

	BPRedrawEntitys redrawElems;
	redrawElems.setDrawMode(BPDrawMode::enTempDraw);
	redrawElems.setDrawPurpose(BPDrawPurpose::enDynamics);
	redrawElems.setDynamicsViews(ev->getViewport());
	redrawElems.doRedraw(m_ptrGraphic->getEntityR());

	//动态标注
	if (m_nStep == 2)
	{
		BPGraphicsPtr ptrDem = __dynamicDimension(ptE, pProject, pModel);
		if (ptrDem.isNull())
			return;
		redrawElems.doRedraw(ptrDem->getEntityR());
	}
}

bool ToolLayoutArcDemo::_onModelMotion(BPBaseButtonEventCP ev)
{
	if (!getDynamicsStarted())
		_beginDynamics();
	return true;
}

bool ToolLayoutArcDemo::_onKeyTransition(bool isDown, ::p3d::platform::P3DVirtualKey key, bool shiftIsDown, bool ctrlIsDown)
{
	if (isDown && key == P3DVirtualKey::enShift)
	{
		m_nType = (m_nType++) % 2;
	}	
	return true;
}

void ToolLayoutArcDemo::__createGraphic(GePoint3d ps, BPProjectP pProject, BPModelP pModel)
{
	//画弧线
	GeEllipse3d ell = GeEllipse3d::createByPointsOnEllipse(m_ptF, ps, m_ptS);
	m_ptC = ell.center;
	IGeCurveBasePtr arc = IGeCurveBase::createEllipse(ell);
	m_ptrGraphic = pModel->createPhysicalGraphics();
	if (m_ptrGraphic == NULL)
		return;
	m_ptrGraphic->addGeCurve(*arc.get());
}

BPGraphicsPtr ToolLayoutArcDemo::__dynamicDimension(GePoint3d ptE, BPProjectP pProject, BPModelP pModel)
{
	//标注样式
	BIMBase::Core::BPDimensionStylePtr dimensionStyle = BIMBase::Core::BPDimensionStyle::create(L"弧线标注样式范例", *pProject);
	if (dimensionStyle.isNull())
		return NULL;
	dimensionStyle->setDimtad(1);
	dimensionStyle->setDimse1(true);
	dimensionStyle->setDimse2(true);
	dimensionStyle->setDimdec(0);
	dimensionStyle->setDimrnd(0);
	dimensionStyle->setDimscale(1);
	dimensionStyle->setDimtxt(200);
	dimensionStyle->setDimasz(500);

	BIMBase::BPColorDef colorDef;
	colorDef.m_rgba.red = 125;
	colorDef.m_rgba.green = 125;
	colorDef.m_rgba.blue = 125;
	UInt32 colorInt = BPColorUtil::getEntityColor(colorDef, *pProject, true);
	dimensionStyle->setDimclrd(colorInt);
	dimensionStyle->replace(L"弧线标注样式范例", pProject);

	//绘制选中对象的标注
	GePoint3d xLine1Point = m_ptF;
	GePoint3d xLine2Point = m_ptS;
	GeVec3d dirVec = GeVec3d::createByStartEndNormalize(xLine1Point, xLine2Point);
	GeVec3d dirVec1 = GeVec3d::createByStartEndNormalize(m_ptF, ptE);

	GeVec3d vOffset = dirVec;
	vOffset.rotate2D(PI / 2);
	vOffset = vOffset * 200;
	GePoint3d midPoint = xLine1Point + vOffset;
	
	//确保起点到终点为逆时针
	GeEllipse3d ell = GeEllipse3d::createByPointsOnEllipse(m_ptF, ptE, m_ptS);
	GeVec3d vec0 = ell.vector0;
	GeVec3d vec90 = ell.vector90;
	double dSweep = ell.sweep;
	GeVec3d cf = GeVec3d::createByStartEnd(m_ptC, m_ptF);
	GeVec3d cs = GeVec3d::createByStartEnd(m_ptC, m_ptS);
	double ang1 = cf.signedAngleTo(cs, GeVec3d::create(0, 0, 1));
	if (ang1 < 0)
		ang1 = ang1 + 2 * PI;

	//弧长标注
	BIMBase::Data::BPDimensionArcLength arclength;
	arclength.setDimstyle(L"弧线标注样式范例");
	//角度标注
	BIMBase::Data::BPDimension3PAngle angledim;
	angledim.setDimstyle(L"弧线标注样式范例");

	if (abs(abs(ang1) - abs(dSweep)) > 0.001)//说明这个弧是顺时针
	{
		arclength.setPoints(m_ptC, midPoint, m_ptS, m_ptF);
		angledim.setPoints(m_ptC, midPoint, m_ptS, m_ptF);
	}
	else
	{
		arclength.setPoints(m_ptC, midPoint, m_ptF, m_ptS);
		angledim.setPoints(m_ptC, midPoint, m_ptF, m_ptS);
	}
	BPGraphicsPtr  ptrGraphics;
	if (m_nType == 0)
		ptrGraphics = angledim.createPhysicalGraphics(*pProject, pProject->getActiveModel()->getModelId(), true);
	else
		ptrGraphics = arclength.createPhysicalGraphics(*pProject, pProject->getActiveModel()->getModelId(), true);
	return ptrGraphics;
}

//对工具进行注册
BPTool* CreateArcTool()
{
	ToolLayoutArcDemo* tool = new ToolLayoutArcDemo();
	return tool;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("layoutArcDemo", &CreateArcTool);
AutoDoRegisterFunctionsEnd