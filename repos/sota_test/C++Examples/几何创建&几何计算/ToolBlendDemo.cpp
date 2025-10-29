#include "stdafx.h"
#include "ToolBlendDemo.h"

using namespace ::p3d::platform;

ToolBlend::ToolBlend()
{
	BPSelectionSetManager::getInstance().emptyAll();  // 清空选择集
	m_eLineType = Straight;
}


ToolBlend::~ToolBlend()
{
	BPSelectionSetManager::getInstance().emptyAll();  // 清空选择集
	getEntityArray()->clear();
}

static void getSinglePathFormCurveVector(GeCurveArrayR path, GeCurveArrayCR loop)
{
	for (int i = 0; i < loop.size(); i++)
	{
		IGeCurveBasePtr ptrCurve = loop.at(i);
		if (IGeCurveBase::CURVE_BASE_TYPE_CurveArray == ptrCurve->getCurveBaseType())
		{
			GeCurveArrayCP pCv = ptrCurve->getChildCurveArrayCP();
			getSinglePathFormCurveVector(path, *pCv);
		}
		else if (IGeCurveBase::CURVE_BASE_TYPE_LineString == ptrCurve->getCurveBaseType())
		{
			pvector<GePoint3d> const* vecLineStringPoints = ptrCurve->getLineStringCP();
			if (vecLineStringPoints == NULL) continue;
			int nPoints = vecLineStringPoints->size();
			if (nPoints <= 0) continue;
			for (int j = 0; j < nPoints - 1; j++)
			{
				GeSegment3d curveSegment;
				ptrCurve->getSegmentInLineString(curveSegment, j);
				if (curveSegment.getLength() <= 0.001) continue;
				IGeCurveBasePtr ptrCurveTemp = IGeCurveBase::createSegment(curveSegment);
				if (ptrCurveTemp.isValid())
					path.push_back(ptrCurveTemp);
			}
		}
		else
		{
			GePoint3d dStart, dEnd;
			ptrCurve->getStartEndPoint(dStart, dEnd);
			path.push_back(ptrCurve);
		}
	}
}

void ToolBlend::_onPostInstall()
{
	T_Super::_onPostInstall();
	_setLocateCursor(true);
	BPSnap::getInstance().enableSnap(true);	
	//添加默认四边形
	{
		BPModelP pModel = BPViewManager::getInstance().getActivedViewport()->getTargetModel();
		if (!pModel)
			return;
		BPGraphicsPtr ptrGraphic = pModel->createPhysicalGraphics();
		if (!ptrGraphic.isValid())
			return;
		GeCurveArrayPtr ptrCurve = GeCurveArray::createRectangle(-2000, -2000, 2000, 2000, -3000, P3D_NAMESPACE_NAME::GeCurveArray::BOUNDARY_TYPE_Outer);
		GeCurveArrayPtr ptrCurve1 = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
		getSinglePathFormCurveVector(*ptrCurve1, *ptrCurve);
		GeRotMatrix matrix = GeRotMatrix::createByAxisAndRotationAngle(2, 30);
		GeTransform trans = GeTransform::create(matrix, GePoint3d::create(0, 0, 3000));
		GeCurveArrayPtr ptrCurve2 = ptrCurve1->clone();
		ptrCurve2->setByTransform(trans);
		GeCurveArrayPtr ptrCurve3 = ptrCurve2->clone();
		ptrCurve3->setByTransform(trans);
		m_vctCurve.push_back(ptrCurve1);
		m_vctCurve.push_back(ptrCurve2);
		m_vctCurve.push_back(ptrCurve3);
		ptrGraphic->addGeCurveArray(*ptrCurve1);
		ptrGraphic->addGeCurveArray(*ptrCurve2);
		ptrGraphic->addGeCurveArray(*ptrCurve3);
		BPEntityId entityId = ptrGraphic->save();
		m_ptrInitEntity = new BPEntity(entityId,*BPProject::getActiveProject());
	}
	
	switch (m_eLineType)
	{
	case Straight:
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"当前使用直线引导线，Shift切换引导线类型，右键确认")); /*，请选择截面轮廓（右键确认）*/
		break;
	case Bspline:
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"当前使用样条曲线引导线，Shift切换引导线类型，右键确认"));
		break;
	default:
		break;
	}
}

void   ToolBlend::_onRestartTool()
{
	ToolBlend* newTool = new ToolBlend();
	newTool->installTool();
}

bool ToolBlend::_onDataButton( BPBaseButtonEventCP ev)
{
	return __super::_onDataButton(ev);	
}

void ToolBlend::_onDynamicFrame( BPBaseButtonEventCP ev)
{
	if (NULL == ev)
		return;
}

::BIMBase::Core::BPEntityPtr ToolBlend::_buildLocateAgenda(BIMBase::Core::BPPickDataCP path, BPBaseButtonEventCP ev)
{
	if (m_vctCurve.size() == 3)
	{
		m_vctCurve.clear();
		m_ptrInitEntity->deleteFromModel();
	}
		
	::BIMBase::Core::BPEntityPtr eh = __super::_buildLocateAgenda(path, ev);
	m_vcEEH.push_back(eh);

	return eh;
}

bool ToolBlend::_onResetButton(BPBaseButtonEventCP ev)
{
	BPViewportP pViewPort = ev->getViewport();
	if (pViewPort == NULL)
		return false;
	BPModelP pModel = pViewPort->getTargetModel();
	if (!pModel)
		return false;

	
	BPGraphicsPtr ptrGraphics = pModel->createPhysicalGraphics();
	if (!ptrGraphics.isValid())
		return false;

	if (m_vcEEH.size() == 3)
	{
		m_vctCurve.clear();
		int nEdegs = -1;
		for (int i = 0; i < m_vcEEH.size(); i++)
		{
			GeCurveArrayPtr ptrOutCurveVector = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
			BPEntityPtr pCurSel = (m_vcEEH.at(i));
			BPEntity eeh = *(pCurSel);
			BIMBase::BPEntityId elementId = eeh.getEntityId();
			::BIMBase::Core::BPGraphicsPtr ptrElePhysicas = BPGraphics::getGraphicsFromEntity(eeh);
			if (ptrElePhysicas.isNull())
				continue;

			for (const BPGraphics::EntryPtr& loadedEntry : *ptrElePhysicas)
			{
				switch (loadedEntry->getType())
				{
				case BPGraphics::Entry::Type::GeCurveArray:
				{
					GeCurveArrayP pCurveVector = loadedEntry->getAsGeCurveArrayP();
					if (pCurveVector == NULL)
						continue;
					GeCurveArrayPtr ptrTemptCurveVector = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
					getSinglePathFormCurveVector(*ptrTemptCurveVector, *pCurveVector);
					for (int i = 0; i < ptrTemptCurveVector->size(); i++)
					{
						ptrOutCurveVector->add(ptrTemptCurveVector->at(i));
					}
				}
				break;
				case BPGraphics::Entry::Type::GeCurveBase:
				{
					IGeCurveBaseP pCurvePrimitiveTemp = loadedEntry->getAsGeCurveBaseP();
					if (pCurvePrimitiveTemp == NULL)
						continue;
					if (pCurvePrimitiveTemp->getCurveBaseType() == IGeCurveBase::CurveBaseType::CURVE_BASE_TYPE_LineString)
					{
						for (int i = 0; i < pCurvePrimitiveTemp->getLineStringP()->size() - 1; i++)
						{
							GeSegment3d seg;
							pCurvePrimitiveTemp->getSegmentInLineString(seg, i);
							ptrOutCurveVector->add(IGeCurveBase::createSegment(seg));
						}
					}
					else
					{
						ptrOutCurveVector->add(pCurvePrimitiveTemp->clone());
					}
				}
				break;
				case BPGraphics::Entry::Type::GeBsplineCurve:
				{
					IGeCurveBaseP pCurvePrimitiveTemp = loadedEntry->getAsGeCurveBaseP();
					if (pCurvePrimitiveTemp != NULL)
						ptrOutCurveVector->add(pCurvePrimitiveTemp->clone());
				}
				break;
				case BPGraphics::Entry::Type::Polyface:
				{
					PolyfaceHandleP pMesh = loadedEntry->getAsPolyfaceHandleP();
					if (nullptr == pMesh)
						return 0;

					vector<GePoint3d > vctTotalPoints;
					TemplateVector<GePoint3d> getpts;
					getpts = pMesh->getPointR();
					for (auto p : getpts)
					{
						vctTotalPoints.push_back(p);
					}

					const int nCnt = vctTotalPoints.size();
					for (int i = 1; i < nCnt; i++)
					{
						GeSegment3d seg = GeSegment3d::create(vctTotalPoints[i - 1], vctTotalPoints[i]);
						IGeCurveBasePtr ptrCurvePrimitiveTemp = IGeCurveBase::createSegment(seg);
						if (ptrCurvePrimitiveTemp.isValid()/* != NULL*/)
							ptrOutCurveVector->add(ptrCurvePrimitiveTemp->clone());
					}

					GeSegment3d seg = GeSegment3d::create(vctTotalPoints[nCnt - 1], vctTotalPoints[0]);
					IGeCurveBasePtr pCurvePrimitiveTemp = IGeCurveBase::createSegment(seg).get();
					if (pCurvePrimitiveTemp.isValid() /*!= NULL*/)
						ptrOutCurveVector->add(pCurvePrimitiveTemp->clone());
				}
				break;
				default:
					break;
				}
			}
			if (ptrOutCurveVector->getNumOfCurveBases() > 0)
			{
				if (nEdegs == -1)
					nEdegs = ptrOutCurveVector->size();
				else if (nEdegs != ptrOutCurveVector->size())
				{
					AfxMessageBox(L"目前不支持不同边数面进行多截面放样！");
					return false;
				}
				m_vctCurve.push_back(ptrOutCurveVector);
			}
		}
	}

	if (m_vctCurve.size() < 2)
	{
		_exitTool();
		return false;
	}

	switch (m_eLineType)
	{
	case ToolBlend::Straight:
	{
		GeRuledSweepInfo ruledSweep(m_vctCurve, true);
		IGeSolidBasePtr ptrSolidSweep = IGeSolidBase::createGeRuledSweep(ruledSweep);
		ptrGraphics->addGeSolidBase(*ptrSolidSweep);
	}
		break;
	case ToolBlend::Bspline:
	{
		createBlend(ptrGraphics, m_vctCurve);
	}
		break;
	default:
		break;
	}	
	ptrGraphics->save();
	for (int i = 0; i < m_vcEEH.size(); i++)
	{
		::p3d::P3DStatus  intt = m_vcEEH[i]->deleteFromModel();
	}
	m_vcEEH.clear();
	if (m_vctCurve.size() == 3)
	{
		m_vctCurve.clear();
		m_ptrInitEntity->deleteFromModel();
	}
	pViewPort->forceHealImmediate();
	return true;
}

bool ToolBlend::_onKeyTransition(bool wentDown, ::p3d::platform::P3DVirtualKey key, bool shiftIsDown, bool ctrlIsDown)
{
	if (key == P3DVirtualKey::enShift && wentDown)
	{
		m_eLineType = LineType(1 - (int)m_eLineType);
		switch (m_eLineType)
		{
		case Straight:
			BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"当前使用直线引导线，右键确认"));
			break;
		case Bspline:
			BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"当前使用样条曲线引导线，右键确认"));
			break;
		default:
			break;
		}
	}
	
	return true;
}

bool ToolBlend::_onModifierKeyTransition(bool wentDown, int key)
{
	return __super::_onModifierKeyTransition(wentDown, key);
}

bool ToolBlend::createBlend(BPGraphicsPtr ptrGraphic, pvector<GeCurveArrayPtr> vctCurves)
{
	int nNum = 120;
	if (vctCurves.size() < 3)
		return false;
	//每个截面离散	
	pvector<pvector<GePoint3d>> disCurvePoints;//每个截面离散点；
	for (auto curve : vctCurves)
	{
		pvector<GePoint3d> vctPoints;		
		for (int i = 0; i < curve->size(); i++)
		{
			int nPoints = 1;
			IGeCurveBasePtr ptrCurveBase = curve->at(i);
			if (ptrCurveBase.isNull())
				continue;
			double dFration = 1.0 / double(nPoints);
			for (int j = 0; j < nPoints; j++)
			{
				GePoint3d point = GePoint3d::createByZero();
				ptrCurveBase->proportToPoint(double(j) * dFration, point);
				vctPoints.push_back(point);
			}
		}
		disCurvePoints.push_back(vctPoints);
	}

	//扫掠路径离散
	pvector<pvector<GePoint3d>> vctSectionPoints;//根据路径离散点整理的截面
	vctSectionPoints.resize(nNum);
	for (int i = 0; i < disCurvePoints[0].size(); i++)
	{
		pvector<GePoint3d> vctPoints;
		BPCurveApproximationFunction::createGeBsplinePoints(vctPoints, pvector<GePoint3d>{disCurvePoints[0][i], disCurvePoints[1][i], disCurvePoints[2][i]}, 3, nNum);
		for (int j = 0; j < nNum; j++)
		{
			vctSectionPoints[j].push_back(vctPoints[j]);
		}
	}

	//创建融合体
	pvector<GeCurveArrayPtr> dispersedCurve;
	for (int j = 0; j < nNum; j++)
	{
		GeCurveArrayPtr ptrCurveDis = GeCurveArray::createLinestringArray(vctSectionPoints[j], P3D_NAMESPACE_NAME::GeCurveArray::BOUNDARY_TYPE_Outer, true);
		dispersedCurve.push_back(ptrCurveDis);
	}
	GeRuledSweepInfo ruledSweep(dispersedCurve, true);
	IGeSolidBasePtr ptrSolidSweep = IGeSolidBase::createGeRuledSweep(ruledSweep);
	ptrGraphic->addGeSolidBase(*ptrSolidSweep);
	return true;
}

BPTool* CreateDemoBlendSelectTool()
{
	ToolBlend* tool = new ToolBlend();
	return tool;
	return NULL;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("blendDemo", &CreateDemoBlendSelectTool);
AutoDoRegisterFunctionsEnd

