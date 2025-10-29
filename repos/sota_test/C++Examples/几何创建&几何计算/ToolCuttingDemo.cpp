#include "stdafx.h"
#include "time.h"
#include "ToolCuttingDemo.h"

using namespace BIMBase;
using namespace BIMBase::FrameWork;
using namespace BIMBase::SolidCore;

ToolCuting::ToolCuting()
{
	m_ptrModelNew = nullptr;
}

ToolCuting::~ToolCuting()
{
	__postTreatment();
}

void ToolCuting::_onPostInstall()
{
	__doCutting();
}

void ToolCuting::__doCutting()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;

	BPViewportP pViewport = BPViewManager::getInstance().getViewport(0);
	if (pViewport == nullptr)
		return;

	BPModelP pModel = pViewport->getTargetModel();
	if (pModel == nullptr)
		return;

	//获取当前model上所有的图素
	BPEntityArray entityArray;
	BPEntityUtil::getEntitiesOfModel(entityArray, *pProject, pModel->getModelId());

	if (entityArray.getCount() == 0)
		return;

	//确定剖切范围
	p3d::pvector<BPGraphicsPtr> pvecGraphics;
	GeRange3d range = GeRange3d::createByNull();
	for (int i = 0; i < entityArray.getCount(); i++)
	{
		GeRange3d range3dew = GeRange3d::createByNull();
		BPEntityPtr ptrCurr = entityArray.getByIndex(i);
		if (!ptrCurr || !ptrCurr.isValid())
			continue;
		PString sClassname;
		ptrCurr->getClassName(sClassname);
		GeTransform tran;
		tran.setByIdentityMatrix();
		BPGraphicsPtr ptrGraphic = BPEntityUtil::transformEntity(*ptrCurr, tran, false);
		pvecGraphics.push_back(ptrGraphic);

		ptrCurr->getRange(range3dew);
		range.extendRange(range3dew);
	}

	BPSymbology symb = BPGraphics::getDefaultSymbology();
	symb.style = 0;
	symb.weight = 1;
	symb.color = BPColorUtil::getEntityColor(RGB(255, 255, 255), *pProject, true);

	BPSymbology symb2 = BPGraphics::getDefaultSymbology();
	symb2.style = 2;
	symb2.weight = 0;
	symb2.color = BPColorUtil::getEntityColor(RGB(255, 255, 255), *pProject, true);

	//剖切范围是所有对象的包围盒
	GePlane3d clipPlane;
	GeTransform sectionBox;
	GeTransform tm;
	__createSectionBoxAndClipPlane(0, range, clipPlane, sectionBox, tm);

	BPHideLineEntity sectionElement(pvecGraphics, clipPlane, true);
	sectionElement.resetSectionBoxOrPlane(&clipPlane, &sectionBox);
	p3d::pmap<size_t, BPClassifyHideLineData> pouqieResult = sectionElement.getClassifyHideLineResult();
	p3d::pmap<size_t, BPClassifyHideLineData>::iterator it = pouqieResult.begin();

	P3DStatus status;
	p3d::Utf8String sModelName = "CuttingModel";
	p3d::platform::P3DModelType modelType = p3d::platform::P3DModelType::enPhysical;
	BIMBase::Core::ModelTreeItemInfo modelTreeItemInfo;
	BPModelPtr ptrNewModel = pProject->createNewModel(status, sModelName, modelType, true, modelTreeItemInfo);
	if (ptrNewModel.isNull())
		return;
	m_ptrModelNew = ptrNewModel;

	for (auto info : pvecGraphics)
	{
		BPGraphicsPtr ptrGraphics = ptrNewModel->createPhysicalGraphics();
		BPGraphicsPtr ptrGraphicsDash = ptrNewModel->createPhysicalGraphics();

		for (; it != pouqieResult.end(); it++)
		{
			BPClassifyHideLineData _classifyCurveDatas = it->second;
			if (_classifyCurveDatas.vctProjectCurve.size() == 0 && _classifyCurveDatas.vctSliceCurve.size() == 0)
				continue;

			for (auto info : _classifyCurveDatas.vctProjectCurve)
			{
				GeCurveArrayPtr ptrCurves = info.m_curve;
				ptrCurves->setByTransform(tm);
				if (info.m_isHidden)
					ptrGraphicsDash->addGeCurveArray(*ptrCurves, symb2);
				else
				{
					ptrGraphics->addGeCurveArray(*ptrCurves, symb);
				}
			}
			for (auto info : _classifyCurveDatas.vctSliceCurve)
			{
				p3d::pvector<BPHideLineData> vetSection = info;
				for (auto infos : vetSection)
				{
					GeCurveArrayPtr ptrCurves = infos.m_curve;
					ptrCurves->setByTransform(tm);
					if (infos.m_isHidden)
						ptrGraphicsDash->addGeCurveArray(*ptrCurves, symb2);
					else
					{
						ptrGraphics->addGeCurveArray(*ptrCurves, symb);
					}
				}
			}
		}

		BPEntityId SLIDE = ptrGraphics->save();
		BPEntityId DASH = ptrGraphicsDash->save();
	}

	PModelId ModelId = ptrNewModel->getModelId();

	//创建view
	CFrameWnd* pView = BIMBase::FrameWork::BPUIFrameWorkUtil::createView();
	if (pView == nullptr)
		return;

	//获取创建view的number
	m_Num = BIMBase::FrameWork::BPUIFrameWorkUtil::getViewNumber(pView);

	//创建的新model在view中显示
	BPViewManager::getInstance().displayModelOnViewPort(ModelId, m_Num);
	BPViewManager::setAllow3DManipulations(m_Num, BPViewManager::BPRotateAxisOption::enRotateNone);
}

void ToolCuting::_onRestartTool()
{
	ToolCuting* newTool = new ToolCuting();
	newTool->installTool();
}

bool ToolCuting::_onDataButton(BPBaseButtonEventCP ev)
{
	return false;
}


bool ToolCuting::_onResetButton(BPBaseButtonEventCP ev)
{
	__postTreatment();
	m_Num = -1;
	_exitTool();
	return false;
}

bool ToolCuting::_onModelMotion(BPBaseButtonEventCP ev)
{
	__super::_onModelMotion(ev);
	if (!getDynamicsStarted())
		_beginDynamics();

	return false;
}

void ToolCuting::_onDynamicFrame(BPBaseButtonEventCP ev)
{
	return;
}


void ToolCuting::__createSectionBoxAndClipPlane(int vecPlane, GeRange3d range, GePlane3d& clipplane, GeTransform& sectionBox, GeTransform& transform)
{
	GeVec3d xydirZ = GeVec3d::create(0,0,0);
	GeVec3d yzdirZ = GeVec3d::create(0, 0, 0);
	GeVec3d	xzdirZ = GeVec3d::create(0, 0, 0);
	BIMBase::Data::BPPlacement placement;

	if (0 == vecPlane)
	{
		//剖切范围是所有对象的包围盒
		//XY平面作为剖切面，剖切平面的法向要与剖切盒子的z轴同向
		GeVec3d dir = GeVec3d::createByStartEndNormalize(range.high, GePoint3d::create(range.high.x, range.high.y, range.low.z));

		GePlane3d xyclipPlane;
		xyclipPlane.setByOriginAndNormal(range.high, dir);

		GeVec3d dirX = GeVec3d::createByStartEnd(range.high, GePoint3d::create(range.low.x, range.high.y, range.high.z));
		GeVec3d dirY = GeVec3d::createByStartEnd(range.high, GePoint3d::create(range.high.x, range.low.y, range.high.z));
		xydirZ = GeVec3d::createByStartEnd(range.high, GePoint3d::create(range.high.x, range.high.y, range.low.z));

		GeTransform xysectionBox = GeTransform::createByOriginAndVectors(range.high, dirX, dirY, xydirZ);
		xydirZ.negate();
		placement.setPlacement(GePoint3d::create(0, 0, 0), xydirZ, 0);
		clipplane = xyclipPlane;
		sectionBox = xysectionBox;
	}
	else if (1 == vecPlane)
	{
		//yz平面作为剖切面,要遵守右手法则，当平面法向为正时，把yz平面的y作为构造剖切盒子的x，z作为构造剖切盒子的y，然后剖切盒子的z要和平面的法向同向
		GeVec3d yzdirX = GeVec3d::create(GePoint3d::create(0, (range.high.y - range.low.y), 0));
		GeVec3d yzdirY = GeVec3d::create(GePoint3d::create(0, 0, (range.high.z - range.low.z)));
		GeVec3d yzdir = GeVec3d::createByStartEndNormalize(range.low, GePoint3d::create(range.high.x, range.low.y, range.low.z));
		yzdirZ = GeVec3d::createByStartEnd(range.low, GePoint3d::create(range.high.x, range.low.y, range.low.z));

		GePlane3d yzclipPlane;
		yzclipPlane.setByOriginAndNormal(range.low, yzdir);
		GeTransform yzsectionBox = GeTransform::createByOriginAndVectors(range.low, yzdirX, yzdirY, yzdirZ);
		//将比如xz,yz这种平面去剖切，要把结果转到xy平面上（因为最终在model上显示的是xy平面的结果，下面做法是比如xz平面，可以理解为xy平面，要把xz平面
		//的y作为xy平面的z）
		yzdirZ.negate();
		placement.setPlacement(GePoint3d::create(0, 0, 0), yzdirZ, 0);
		clipplane = yzclipPlane;
		sectionBox = yzsectionBox;
	}
	else
	{
		//xz平面作为剖切面,要遵守右手法则，当平面法向为正时，把xz平面的z作为构造剖切盒子的x，x作为构造剖切盒子的y，然后剖切盒子的z要和平面的法向同向
	    //当平面法向量为负时，把xz平面的x作为构造剖切盒子的x，z作为构造剖切盒子的y,这里以法向为正为例
		GeVec3d xzdirX = GeVec3d::create(GePoint3d::create(0, 0, (range.high.z - range.low.z)));
		GeVec3d xzdirY = GeVec3d::create(GePoint3d::create((range.high.x - range.low.x), 0, 0));
		GeVec3d xzdir = GeVec3d::createByStartEndNormalize(range.low, GePoint3d::create(range.low.x, range.high.y, range.low.z));
		xzdirZ = GeVec3d::createByStartEnd(range.low, GePoint3d::create(range.low.x, range.high.y, range.low.z));

		GePlane3d xzclipPlane;
		xzclipPlane.setByOriginAndNormal(range.low, xzdir);
		GeTransform xzsectionBox = GeTransform::createByOriginAndVectors(range.low, xzdirX, xzdirY, xzdirZ);
		xzdirZ.negate();
		placement.setPlacement(GePoint3d::create(0, 0, 0), xzdirZ, 0);
		clipplane = xzclipPlane;
		sectionBox = xzsectionBox;
	}

	transform = placement.toTransform();
	transform.setByInverse(transform);
}

void ToolCuting::__postTreatment()
{
	//关闭新建的视图
	CFrameWnd* pView = BIMBase::FrameWork::BPUIFrameWorkUtil::getView(m_Num);
	if (pView != nullptr)
	{
		pView->SendMessage(WM_CLOSE); //post message
	}

	if (m_ptrModelNew.isValid())
		m_ptrModelNew->emptyContent();
}

void registerCutDrawingTool()
{
	ToolCuting* tool = new ToolCuting();
	tool->installTool();
}
AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("cuttingDemo", &registerCutDrawingTool);
AutoDoRegisterFunctionsEnd