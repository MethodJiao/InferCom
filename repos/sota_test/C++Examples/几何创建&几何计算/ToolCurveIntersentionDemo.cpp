#include "stdafx.h"
#include "ToolCurveIntersentionDemo.h"

GeCurveArrayPtr g_ptrCurveOne = nullptr;

ToolCurveIntersentionDemo::ToolCurveIntersentionDemo()
{
}

ToolCurveIntersentionDemo::~ToolCurveIntersentionDemo()
{

}

void ToolCurveIntersentionDemo::_onPostInstall()
{
	if (g_ptrCurveOne == nullptr)
		g_ptrCurveOne = __createCurve();

	BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请绘制曲线，右键完成绘制"));
}

void ToolCurveIntersentionDemo::_onRestartTool()
{

}

bool ToolCurveIntersentionDemo::_onDataButton(BPBaseButtonEventCP ev)
{
	m_vctPts.push_back(*ev->getPoint());
	return true;
}

bool ToolCurveIntersentionDemo::_onResetButton(BPBaseButtonEventCP)
{
	if (m_vctPts.size() == 0)
	{
		_exitTool();
		return true;
	}

	IGeCurveBasePtr ptrLine = IGeCurveBase::createLineString(m_vctPts);
	if (ptrLine.isNull())
		return false;

	GeCurveArrayPtr ptrCurve = GeCurveArray::create(GeCurveArray::BoundaryType::BOUNDARY_TYPE_Open, ptrLine);
	if (ptrCurve.isNull())
		return false;

	BPModelP pModel = BPViewManager::getInstance().getActivedViewport()->getTargetModel();
	if (pModel == nullptr)
		return false;

	BPGraphicsPtr ptrGraphic = pModel->createPhysicalGraphics();
	ptrGraphic->addGeCurve(*ptrLine.get());

	ptrGraphic->save();

	//求交
	__calculateIntersention(g_ptrCurveOne, ptrCurve);
	m_vctPts.clear();

	BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请绘制曲线，右键完成绘制"));
	return true;
}

void ToolCurveIntersentionDemo::_onDynamicFrame(BPBaseButtonEventCP ev)
{
	if (m_vctPts.size() == 0)
		return;

	pvector<GePoint3d> vctPts;
	vctPts.insert(vctPts.end(),m_vctPts.begin(), m_vctPts.end());

	vctPts.push_back(*ev->getPoint());
	IGeCurveBasePtr ptrLine = IGeCurveBase::createLineString(vctPts);
	if (ptrLine.isNull())
		return;


	BPModelP pModel = BPViewManager::getInstance().getActivedViewport()->getTargetModel();
	if (pModel == nullptr)
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

bool ToolCurveIntersentionDemo::_onModelMotion(BPBaseButtonEventCP ev)
{
	if (!getDynamicsStarted())
		_beginDynamics();
	return true;
}


GeCurveArrayPtr ToolCurveIntersentionDemo::__createCurve()
{
	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return nullptr;
	BPProjectPtr ptrProject = pProjectManager->getMainProject();
	if (ptrProject.isNull())
		return nullptr;
	BPModelBaseP pModel = pProjectManager->getActiveModel();
	if (pModel == nullptr)
		return nullptr;

	BPGraphicsPtr ptrPhysical = pModel->createPhysicalGraphics();
	if (ptrPhysical.isNull())
		return nullptr;

	GePoint3d pt0 = GePoint3d::create(1000, 2000, 0);
	GePoint3d pt1 = GePoint3d::create(1500, 3000, 0);
	GePoint3d pt2 = GePoint3d::create(3000, 4000, 0);
	GePoint3d pt3 = GePoint3d::create(4000, 2000, 0);
	GePoint3d pt4 = GePoint3d::create(3500, 1000, 0);
	GePoint3d pt5 = GePoint3d::create(5000, -500, 0);
	GePoint3d pt6 = GePoint3d::create(2000, -800, 0);

	GeCurveArrayPtr ptrCurveA = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Open);
	if (ptrCurveA.isNull())
		return nullptr;

	IGeCurveBasePtr ptrLine = IGeCurveBase::createSegment(GeSegment3d::create(pt0, pt1));
	if (ptrLine.isNull())
		return nullptr;
	ptrCurveA->add(ptrLine);

	IGeCurveBasePtr ptrEllip1 = IGeCurveBase::createEllipse(GeEllipse3d::createByPointsOnEllipse(pt1, pt2, pt3));
	if (ptrEllip1.isNull())
		return nullptr;
	ptrCurveA->add(ptrEllip1);

	IGeCurveBasePtr ptrEllip2 = IGeCurveBase::createEllipse(GeEllipse3d::createByPointsOnEllipse(pt3, pt4, pt5));
	if (ptrEllip2.isNull())
		return nullptr;
	ptrCurveA->add(ptrEllip2);

	IGeCurveBasePtr ptrLine2 = IGeCurveBase::createSegment(GeSegment3d::create(pt5, pt6));
	if (ptrLine2.isNull())
		return nullptr;
	ptrCurveA->add(ptrLine2);

	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return nullptr;

	BPSymbology symb;
	symb.style = 0;  //线型
	symb.weight = 8;  //线宽
	symb.color = BPColorUtil::getEntityColor(RGB(255, 0, 0), *pProject, true);
	ptrPhysical->addGeCurveArray(*ptrCurveA.get(), symb);
	
	ptrPhysical->save();

	return ptrCurveA;
}

void ToolCurveIntersentionDemo::__calculateIntersention(GeCurveArrayPtr curveA, GeCurveArrayPtr curveB)
{
	GeCurveArrayPtr ptrIntersectionA = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Open);
	GeCurveArrayPtr ptrIntersectionB = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Open);
	GeMatrix4d pWorldToLocal;
	pWorldToLocal.setByIdentityMatrix();
	GeCurveFunction::calculateIntersention2D(*ptrIntersectionA, *ptrIntersectionB, *curveA, *curveB, &pWorldToLocal);
	int nPtNumA = (int)ptrIntersectionA->size();
	int nPtNumB = (int)ptrIntersectionB->size();
	double dDot = 0.0001;
	vector<GePoint3d> intersA;
	if (ptrIntersectionA->size() > 0 && ptrIntersectionB->size() > 0)     // 有交点
	{
		for (size_t i = 0; i < ptrIntersectionA->size(); i++)
		{
			GePoint3d pointA;


			GeProportCurveInfoCP pDetailA = ptrIntersectionA->at(i)->getProportCurveInfoCP();

			if (pDetailA->m_proportPara0 >= -dDot && pDetailA->m_proportPara0 <= 1 + dDot)
			{
				pDetailA->m_parentCurve->proportToPoint(pDetailA->m_proportPara0, pointA);
				intersA.push_back(pointA);
			}
		}

	}

	vector<GePoint3d> inters;
	GeCurveLocationInfo closePoint;
	int n = 1;
	CString sPt = _T("");
	for (int i = 0; i < intersA.size(); i++)
	{
		GePoint3d nPtA = intersA[i];

		curveB->getClosestPointBounded(nPtA, closePoint);
		double fr2 = closePoint.m_proportPara;

		if (fr2 >= -dDot && fr2 <= 1 + dDot)
		{
			inters.push_back(nPtA);
			sPt.AppendFormat(_T("交点%d: x=%f ,y=%f ,z=%f\n"), n, nPtA.x, nPtA.y, nPtA.z);
			n++;
		}
	}

	CString str = _T("");
	str.Format(_T("两条曲线有%d个交点\n"), inters.size());
	AfxMessageBox(str + sPt);


}

//对工具进行注册
BPTool* CreateCurveIntersentionDemoTool()
{
	ToolCurveIntersentionDemo* tool = new ToolCurveIntersentionDemo();
	return tool;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("curveIntersentionDemo", &CreateCurveIntersentionDemoTool);
AutoDoRegisterFunctionsEnd