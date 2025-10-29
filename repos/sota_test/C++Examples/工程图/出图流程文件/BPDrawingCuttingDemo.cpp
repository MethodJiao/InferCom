#include "stdafx.h"
#include "BPDrawingCuttingDemo.h"
#include "BPDrawingInfoDemo.h"
#include "BPCutModelManagerDemo.h"


using namespace DemoObject;
UInt32 newView = 0;
BPDrawingCuttingDemo::BPDrawingCuttingDemo()
{

}

BPDrawingCuttingDemo::~BPDrawingCuttingDemo()
{

}

BPDrawingCuttingDemoR BPDrawingCuttingDemo::Get()
{
	static BPDrawingCuttingDemo single;
	return single;
}
void BPDrawingCuttingDemo::getAllModelRange(BPProjectP pProject,GeRange3d& range)
{
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
	for (int i = 0; i < entityArray.getCount(); i++)
	{
		GeRange3d range3dew = GeRange3d::createByNull();
		BPEntityPtr ptrCurr = entityArray.getByIndex(i);
		if (!ptrCurr || !ptrCurr.isValid())
			continue;
		
		GeTransform tran;
		tran.setByIdentityMatrix();
		BPGraphicsPtr ptrGraphic = BPEntityUtil::transformEntity(*ptrCurr, tran, false);
		pvecGraphics.push_back(ptrGraphic);
		ptrCurr->getRange(range3dew);
		range.extendRange(range3dew);
	}
}
void BPDrawingCuttingDemo::getPhysicalModelElements(BPProjectP pProject, BPModelP model, p3d::pvector<pair<BPDataKey, BPGraphicsPtr>>& cutinstance)
{
	if (pProject == nullptr || model == nullptr)
		return;

	//获取当前model上所有的图素
	BPEntityArray entityArray;
	BPEntityUtil::getEntitiesOfModel(entityArray, *pProject, model->getModelId());

	if (entityArray.getCount() == 0)
		return;

	//确定剖切范围
	p3d::pvector<BPGraphicsPtr> pvecGraphics;
	for (int i = 0; i < entityArray.getCount(); i++)
	{
		BPEntityPtr ptrCurr = entityArray.getByIndex(i);
		if (!ptrCurr || !ptrCurr.isValid())
			continue;
		BPDataKey datakey = BPDataUtil::getDataKeyOnEntity(*ptrCurr);
		GeTransform tran;
		tran.setByIdentityMatrix();
		BPGraphicsPtr ptrGraphic = BPEntityUtil::transformEntity(*ptrCurr, tran, false);
		pvecGraphics.push_back(ptrGraphic);
		cutinstance.push_back(make_pair(datakey, ptrGraphic));

	}

}
void BPDrawingCuttingDemo::cutting(PBBimCore::PBModelInfoPtr modelInfoPtr/*CString drawingModelName*/,p3d::pvector<pair<BPDataKey, BPGraphicsPtr>> cutinstance, GePlane3d clipPlane, GeTransform sectionBox, GeTransform tm, BPProjectP pProject)
{
	if (pProject == nullptr)
		return;
	P3DStatus status;
	
	BPModelP ptrNewModel = pProject->loadModelById(modelInfoPtr->GetModelId());
	if (ptrNewModel == nullptr)
		return;
	p3d::pvector<pair<BPDataKey, BPGraphicsPtr>>::iterator iteruse = cutinstance.begin();
	int si = cutinstance.size();
	//消隐
	pvector<BPGraphicsPtr> pvecGraphics;
	std::map<size_t, BPDataKey> cutbindkey;
	size_t bindcou = 0;
	for (iteruse;iteruse != cutinstance.end();iteruse++)
	{
		auto gra = iteruse->second;
		auto key = iteruse->first;
		cutbindkey.insert(make_pair(bindcou, key));
		bindcou++;
		pvecGraphics.push_back(gra);
	}
	
	BPHideLineEntity sectionElement(pvecGraphics, clipPlane, true);
	sectionElement.resetSectionBoxOrPlane(&clipPlane, &sectionBox);
	p3d::pmap<size_t, BPClassifyHideLineData> pouqieResult = sectionElement.getClassifyHideLineResult();
	p3d::pmap<size_t, BPClassifyHideLineData>::iterator it = pouqieResult.begin();

	BPSymbology symb = BPGraphics::getDefaultSymbology();
	symb.style = 0;
	symb.weight = 1;
	symb.color = BPColorUtil::getEntityColor(RGB(255, 255, 255), *pProject, true);

	BPSymbology symb2 = BPGraphics::getDefaultSymbology();
	symb2.style = 2;
	symb2.weight = 0;
	symb2.color = BPColorUtil::getEntityColor(RGB(255, 255, 255), *pProject, true);

	

	for (; it != pouqieResult.end(); it++)
	{
		BPGraphicsPtr ptrGraphics = ptrNewModel->createPhysicalGraphics();
		BPGraphicsPtr ptrGraphicsDash = ptrNewModel->createPhysicalGraphics();
		size_t cutnum = it->first;
		std::map<size_t, BPDataKey>::iterator  iterkey = cutbindkey.find(cutnum);
		if(iterkey == cutbindkey.end())
			continue;

		BPClassifyHideLineData _classifyCurveDatas = it->second;
		if (_classifyCurveDatas.vctProjectCurve.size() == 0 && _classifyCurveDatas.vctSliceCurve.size() == 0)
			continue;

		for (auto info : _classifyCurveDatas.vctProjectCurve)
		{
			
			GeCurveArrayPtr ptrCurves = info.m_curve;
			if (!ptrCurves.isValid())
				continue;
			
			if (info.m_isHidden || info.m_abandoned)
				continue;
			ptrCurves->setByTransform(tm);
			ptrGraphics->addGeCurveArray(*ptrCurves, symb);
		}
		for (auto info : _classifyCurveDatas.vctSliceCurve)
		{
			p3d::pvector<BPHideLineData> vetSection = info;
			for (auto infos : vetSection)
			{
				if (!infos.m_curve.isValid())
					continue;
				if (infos.m_isHidden || infos.m_abandoned)
					continue;
				GeCurveArrayPtr ptrCurves = infos.m_curve;
				ptrCurves->setByTransform(tm);
				ptrGraphics->addGeCurveArray(*ptrCurves, symb);
			}
		}

		BPEntityId SLIDE = ptrGraphics->save();
		BPEntityId DASH = ptrGraphicsDash->save();
		//通过datamanager是否要二三维联动来确定是否进行绑定
		//如果需要绑定
		BPEntity slideenti(SLIDE, *ptrNewModel);
		BPEntity projectenti(DASH, *ptrNewModel);

		if (slideenti.isValid())
		{
			BPDataUtil::bindEntityToData(SLIDE, iterkey->second/*iteruse->first*/, pProject);
		}

		if (projectenti.isValid())
		{
			BPDataUtil::bindEntityToData(DASH, iterkey->second/*iteruse->first*/, pProject);
		}
	}

	PModelId ModelId = ptrNewModel->getModelId();
	BPEntityArray elements;
	BPEntityUtil::getEntitiesOfModel(elements,*pProject, ModelId);
	GeRange3d cutmodelrange = GeRange3d::createByNull();
	for (int i = 0; i < elements.getCount();i++)
	{
		BPEntityPtr curr = elements.getByIndex(i);
		if (!curr || !curr.isValid())
			continue;
		GeRange3d range = GeRange3d::createByNull();
		curr->getRange(range);
		cutmodelrange.extendRange(range);
	}

	vector<int> activeViewSet;
	BPViewManager::getInstance().getAllActiveViewports(activeViewSet);
	bool ismutiviews = activeViewSet.size() == 1 ? false : true;
	if (!ismutiviews)
	{
		BIMBase::BPUserInputManager::exeCommand("view_style_OPEN_NEW");
		newView = BPViewManager::getInstance().getActiveIndex();
	}

	//创建的新model在view中显示
	BPViewManager::getInstance().displayModelOnViewPort(ModelId, newView);
	BPViewManager::setAllow3DManipulations(newView, BPViewManager::BPRotateAxisOption::enRotateNone);
}

void BPDrawingCuttingDemo::__createSectionBoxAndClipPlane(int vecPlane, GeRange3d range, GePlane3d& clipplane, GeTransform& sectionBox, GeTransform& transform)
{
	GeVec3d xydirZ = GeVec3d::create(0, 0, 0);
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


PBBimCore::PBModelInfoPtr  BPDrawingCuttingDemo::getModelInfo(PString sName)
{
	BPProjectPtr project = BPProject::getMainProject();
	if (project == nullptr)
		return nullptr;
	T_PBModelInfoPtrArrayCR modelInfoArray = PBModelInfoManager::Get().GetAllModels();
	for (PBModelInfoPtr modelInfoPtr : modelInfoArray)
	{
		if (modelInfoPtr == nullptr)
			continue;
		PModelId id = modelInfoPtr->GetModelId();
		PString name = modelInfoPtr->GetDisplayedName();
		if (sName == name )//说明这个图纸已经生成了
		{
			P3DModelUtil::DeleteElementsInModel(*project,id,true);
			return modelInfoPtr;
		}
	}
	CString sguid = PString(BPDataUtil::generateGuidString().c_str()).c_str();
	BIMBase::Core::ModelTreeItemInfo modelTreeItemInfo;
	PBStoreyModelViewInfoPtr modelview = PBStoreyModelViewManager::Get().createSheet(*project, modelTreeItemInfo);
	if (modelview == nullptr)
		return nullptr;
	PBSheetModelInfoP sheetmodel = dynamic_cast<PBSheetModelInfoP>(modelview->GetModelInfo().get());
	if(sheetmodel == nullptr)
		return nullptr;
	sheetmodel->SetScale(100);
	sheetmodel->SetDisplayedName(sName);
	sheetmodel->SetModelType(PBModelType::Sheet);
	sheetmodel->replaceInProject(*project);
	return sheetmodel;
}

void BPDrawingCuttingDemo::addElementToCut(BPProjectP pProject, BPModelP model, BPEntity entity, PBBimCore::PBModelInfoPtr drawmodelInfoPtr, BPDrawingParasManagerDemo::eDrawingview type)
{
	if (pProject == nullptr || model == nullptr || drawmodelInfoPtr == nullptr)
		return;
	GeTransform tm = GeTransform::createIdentityMatrix();
	pvector<BPDataKey> dataKey;//存现在三维图上所有的entity，然后再一次判断哪些需要剖切，哪些需要符号化
	p3d::pvector<pair<BPDataKey, BPGraphicsPtr>> cutinstance;
	//符号化的数据绑定
	p3d::pvector<pair<BPDataKey, BPGraphicsPtr>>cutinstancesym;//做数据绑定
	BPModelP drawmodel = pProject->loadModelById(drawmodelInfoPtr->GetModelId());
	BPDrawingCuttingDemo::Get().getPhysicalModelElements(pProject, model, cutinstance);

	if(!entity.isValid())
		return;
	IBPObjectPtr ptrPbObj = BPObjectExtensionManager::getInstance().getBPObject(entity);
	if (!ptrPbObj.isValid())
		return;
	BPDemoGraphicElementDemoP pgraphicele = dynamic_cast<BPDemoGraphicElementDemoP>(ptrPbObj.get());
	
	p3d::pvector<pair<BPDataKey, BPGraphicsPtr>>::iterator iteruse = cutinstance.begin();
	for (iteruse;iteruse != cutinstance.end();iteruse++)
	{
		auto keys = iteruse->first;
		dataKey.push_back(keys);//所有3维
	}
	cutinstance.clear();
	for (auto key : dataKey)
	{
		
		BPEntityId entityid = 	BPEntityUtil::getPrimaryEntityWithData(*pProject,key,model->getModelId());
		BPEntity entity(entityid, *pProject);
		if (!entity.isValid())
			continue;
		IBPObjectPtr ptrPbObj = BPObjectExtensionManager::getInstance().getBPObject(entity);
		if (!ptrPbObj.isValid())
			return;
		BPDemoGraphicElementDemoP pgraphicele = dynamic_cast<BPDemoGraphicElementDemoP>(ptrPbObj.get());
		if (pgraphicele != NULL)
		{
			BPGraphicsPtr gra = pgraphicele->createPhysicalGraphicsForDrawing(*pProject, model->getModelId());
			if (gra == nullptr)//说明需要剖切
			{
				BPGraphicsPtr ptrGraphic = BPEntityUtil::transformEntity(entity, tm, false);
				if (ptrGraphic != nullptr)
				{
					cutinstance.push_back(make_pair(key, ptrGraphic));
				}
			}
			else//说明需要符号化
			{
				BPGraphicsUtils::transformPhysicalGraphics(*gra, pgraphicele->getPlacement().toTransform());
				cutinstancesym.push_back(make_pair(key, gra));
			}
		}

	}

	GeRange3d range = GeRange3d::createByNull();

	getAllModelRange(pProject, range);

	GePlane3d clipPlane;
	GeTransform sectionBox;
	
	//假设视图参数设置的事xy平面
	int types = 0;
	if (type == BPDrawingParasManagerDemo::eDrawingview::X_Y)
		types = 0;
	else if (type == BPDrawingParasManagerDemo::eDrawingview::Y_Z)
		types = 1;
	else if (type == BPDrawingParasManagerDemo::eDrawingview::X_Z)
		types = 2;

	__createSectionBoxAndClipPlane(types, range, clipPlane, sectionBox, tm);
	
	PModelId drawid = drawmodelInfoPtr->GetModelId();
	
	P3DModelUtil::DeleteElementsInModel(*pProject, drawid, true);
	BPEntityArray entityArray2;
	BPEntityUtil::getEntitiesOfModel(entityArray2, *pProject, drawmodelInfoPtr->GetModelId());

	int si1 = entityArray2.getCount();
	if (cutinstance.size() != 0)
	{
		cutting(drawmodelInfoPtr, cutinstance, clipPlane, sectionBox, tm, pProject);

	}

	if (cutinstancesym.size() != 0)
	{
		BPModelP ptrNewModel = pProject->loadModelById(drawmodelInfoPtr->GetModelId());
		if (ptrNewModel != nullptr)
		{

			p3d::pvector<pair<BPDataKey, BPGraphicsPtr>>::iterator iteruse = cutinstancesym.begin();
			for (iteruse;iteruse != cutinstancesym.end();iteruse++)
			{
				auto gra = iteruse->second;
				gra->setModel(ptrNewModel);

				BPEntityId entityid = gra->save();
				BPEntity enti(entityid, *ptrNewModel);

				if (enti.isValid())
				{
					BPDataUtil::bindEntityToData(entityid, iteruse->first, pProject);
				}

			}
			vector<int> activeViewSet;
			BPViewManager::getInstance().getAllActiveViewports(activeViewSet);
			bool ismutiviews = activeViewSet.size() == 1 ? false : true;
			if (!ismutiviews)
			{
				BIMBase::BPUserInputManager::exeCommand("view_style_OPEN_NEW");
				newView = BPViewManager::getInstance().getActiveIndex();
			}


			//创建的新model在view中显示
			BPViewManager::getInstance().displayModelOnViewPort(ptrNewModel->getModelId(), newView);
			BPViewManager::setAllow3DManipulations(newView, BPViewManager::BPRotateAxisOption::enRotateNone);
		}

	}
	PString name = drawmodelInfoPtr->GetDisplayedName();
	BPCutModelManagerDemo::Get().addModel(name.c_str(), drawmodelInfoPtr);
	
	BPEntityArray entityArray;
	BPEntityUtil::getEntitiesOfModel(entityArray, *pProject, drawmodelInfoPtr->GetModelId());
	BPDrawingInfoDemo::Get().drawDimension(drawmodelInfoPtr);

}