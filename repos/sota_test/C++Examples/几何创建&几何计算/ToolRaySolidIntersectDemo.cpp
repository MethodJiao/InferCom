#include "stdafx.h"
#include "ToolRaySolidIntersectDemo.h"


IGeSolidBasePtr g_ptrSolid = nullptr;
ToolRaySolidIntersectDemo::ToolRaySolidIntersectDemo()
{

}

ToolRaySolidIntersectDemo::~ToolRaySolidIntersectDemo()
{

}

void ToolRaySolidIntersectDemo::_onPostInstall()
{
	BPTempUcsMngr::deactivate();

	if(g_ptrSolid == nullptr)
		g_ptrSolid = __getSolid();

	BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请绘制射线"));
}

void ToolRaySolidIntersectDemo::_onRestartTool()
{

}

bool ToolRaySolidIntersectDemo::_onDataButton(BPBaseButtonEventCP ev)
{
	if (ev == nullptr)
		return false;

	m_ptClick.push_back(*ev->getPoint());

	if (m_ptClick.size() != 2)
		return false;

	GePoint3d ptS = m_ptClick[0];
	GePoint3d ptE = m_ptClick[1];

	m_ptClick.clear();

	if (g_ptrSolid.isNull())
		return false;

	GeRay3d ray = GeRay3d::createByOriginAndVector(ptS, GeVec3d::createByStartEndNormalize(ptS, ptE));
	
	pvector<GeSolidLocationInfo> pickData;
	g_ptrSolid->addRayIntersect(pickData, ray);

	if (pickData.size() == 0)
	{
		AfxMessageBox(_T("绘制的射线与实体没有相交点"));
		return false;
	}

	double disMax = 0;
	GePoint3d pt = GePoint3d::createByZero();
	CString sPt = _T("");

	int n = 1;
	for (auto info:pickData)
	{
		GePoint3d ptGet = info.getPoint();
		__drawSphere(ptGet);
		sPt.AppendFormat(_T("交点%d: x=%f ,y=%f ,z=%f\n"), n,ptGet.x, ptGet.y, ptGet.z);
		if (ptGet.distance(ptS) > disMax)
		{
			disMax = ptGet.distance(ptS);
			pt = ptGet;
		}	
		n++;
	}

	if (disMax == 0)
		return false;

	ptE = pt + GeVec3d::createByStartEndNormalize(ptS, ptE) * 500;

	IGeCurveBasePtr ptrLine = IGeCurveBase::createSegment(GeSegment3d::create(ptS, ptE));
	BPModelP pModel = BPViewManager::getInstance().getActivedViewport()->getTargetModel();
	if (pModel == nullptr)
		return false;

	BPGraphicsPtr ptrGraphic = pModel->createPhysicalGraphics();
	ptrGraphic->addGeCurve(*ptrLine);
	ptrGraphic->save();

	CString str = _T("");
	str.Format(_T("射线与实体有%d个交点\n"), pickData.size());
	AfxMessageBox(str+ sPt);

	BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请绘制射线"));
	return true;
}

bool ToolRaySolidIntersectDemo::_onResetButton(BPBaseButtonEventCP)
{
	_exitTool();
	return true;
}

void ToolRaySolidIntersectDemo::_onDynamicFrame(BPBaseButtonEventCP ev)
{
	if (m_ptClick.size() == 0)
		return;

	BPModelP pModel = BPViewManager::getInstance().getActivedViewport()->getTargetModel();
	if (pModel == nullptr)
		return;

	GePoint3d ptE = *ev->getPoint();
	IGeCurveBasePtr ptrLine = IGeCurveBase::createSegment(GeSegment3d::create(m_ptClick[0], ptE));
	if (ptrLine.isNull())
		return;

	BPGraphicsPtr ptrGraphic = pModel->createPhysicalGraphics();
	ptrGraphic->addGeCurve(*ptrLine.get());
	if (SUCCESS != ptrGraphic->finish())
		return;

	BPRedrawEntitys redrawElems;
	redrawElems.setDrawMode(BPDrawMode::enTempDraw);
	redrawElems.setDrawPurpose(BPDrawPurpose::enDynamics);
	redrawElems.setDynamicsViews(ev->getViewport());
	redrawElems.doRedraw(ptrGraphic->getEntityR());
}

bool ToolRaySolidIntersectDemo::_onModelMotion(BPBaseButtonEventCP ev)
{
	if (!getDynamicsStarted())
		_beginDynamics();
	return true;
}


IGeSolidBasePtr ToolRaySolidIntersectDemo::__getSolid()
{
	IGeSolidBasePtr extrusion = nullptr;

	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return extrusion;
	BPProjectPtr ptrProject = pProjectManager->getMainProject();
	if (ptrProject.isNull())
		return extrusion;

	BPModelBaseP pModel = ptrProject->getActiveModel();
	if (pModel == nullptr)
		return extrusion;

	BPGraphicsPtr ptrPhysical = pModel->createPhysicalGraphics();
	if (ptrPhysical.isNull())
		return extrusion;

	int nWidth = 2000;
	int nLength = 2000;
	int nHeight = 5000;

	//绘制底部外轮廓
	GeCurveArrayPtr ptrOutLines = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
	pvector<GePoint3d> pts;
	IGeCurveBasePtr ptrLine;

	pts.push_back(GePoint3d::create(0, -nWidth / 2, -500));
	pts.push_back(GePoint3d::create(nLength, -nWidth / 2, -500));
	pts.push_back(GePoint3d::create(nLength, nWidth / 2, -500));
	pts.push_back(GePoint3d::create(0, nWidth / 2, -500));
	pts.push_back(GePoint3d::create(0, -nWidth / 2, -500));
	ptrLine = IGeCurveBase::createLineString(pts);
	ptrOutLines->push_back(ptrLine);

	//向Z方向拉伸
	GeVec3d veccc = GeVec3d::create(0, 0, nHeight);
	GeExtrusionInfo extrData(ptrOutLines, veccc, true);
	extrusion = IGeSolidBase::createGeExtrusion(extrData);

	ptrPhysical->addGeSolidBase(*extrusion);

	ptrPhysical->save();
	return extrusion;
}

void ToolRaySolidIntersectDemo::__drawSphere(GePoint3d pt)
{
	BPModelPtr ptrModel = BPModel::getActiveModel();
	if (ptrModel.isNull())
		return;

	BIMBase::Core::BPProjectP pProject = BIMBase::Core::BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == NULL)
		return;

	BPSymbology symb;
	symb.style = 0;  //线型
	symb.weight = 0;  //线宽
	symb.color = BPColorUtil::getEntityColor(RGB(255, 0, 0), *pProject, true);

	BPGraphicsPtr ptrPhysical = ptrModel->createPhysicalGraphics();
	if (ptrPhysical.isNull())
		return;

	GeSphereInfo sphere(pt, 200);
	IGeSolidBasePtr ptrExtrusion = IGeSolidBase::createGeSphere(sphere);
	ptrPhysical->addGeSolidBase(*ptrExtrusion, symb);
	ptrPhysical->save();

}

//对工具进行注册
BPTool* CreateRaySolidIntersectDemoTool()
{
	ToolRaySolidIntersectDemo* tool = new ToolRaySolidIntersectDemo();
	return tool;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("raySolidIntersectDemo", &CreateRaySolidIntersectDemoTool);
AutoDoRegisterFunctionsEnd