#include "stdafx.h"

void HelloBIMBase()
{
	AfxMessageBox(L"Hello BIMBase！");
}

void createGroup()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	BPModelBaseP pModel = pProject->getActiveModel();
	p3d::pset<BPEntityId> entitySet;
	vector<BPDataKey>  vctKey;
	//获取当前model所有构件
	BPEntityUtil::getEntitiesOfModel(entitySet, *pProject, pModel->getModelId());
	for (auto entityId : entitySet)
	{
		BPEntity entity(entityId, *BPProject::getActiveProject());
		BPDataKey key = BPDataUtil::getDataKeyOnEntity(entity);
		if (key.isValid())
			vctKey.push_back(key);
	}

	//创建组合
	BPGroupUtil::group(vctKey, L"组合Demo");
	//设置组合状态为开始组
	BPGroupManager::getInstance().setGroupMode(BPGroupMode::enGroup);
}

void crossFileToCopy()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	//需要选中需要复制的构件
	if (BPSelectionSetManager::getInstance().getCount() == 0)
		return;

	BIMBase::Core::BPEntityPtr ptrEntity = BIMBase::Core::BPSelectionSetManager::getInstance().getEntityByIndex(0);
	if (ptrEntity.isNull())
		return;

	BIMBase::Core::BPDataPtr ptrData = BPDataUtil::getDataOnEntity(*ptrEntity);
	if (!ptrData.isValid())
		return;

	IBPObjectPtr ptrObjCopy = BPObjectExtensionManager::getInstance().getBPObject(*pProject, ptrData->getDataKey());
	if (ptrObjCopy.isNull())
		return;
	//打开另外一个工程的路径，需要自己设置
	p3d::PString docPath = L"D:\\project\\test1.p3d";
	p3d::P3DStatus status;
	
	BIMBase::Core::BPProjectP _project = BPApplication::getInstance().getProjectManager()->openProject(status, docPath, BIMBase::Core::BPProject::BPOpenMode::enReadWrite);
	if (_project == nullptr)
		return;

	PModelId modelId = _project->findModelIdByName("SYSTEM_AXIS");

	::p3d::P3DStatus sta = ptrObjCopy->addToProject(*_project, modelId);
	p3d::P3DStatus staSave = BPApplication::getInstance().getProjectManager()->saveProject(_project->getProjectHandle());
}

void createComponent()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	BPModelBaseP pModel = pProject->getActiveModel();
	p3d::pset<BPEntityId> entitySet;
	pvector<BPDataKey>  vctKey;
	//获取当前model所有构件
	BPEntityUtil::getEntitiesOfModel(entitySet, *pProject, pModel->getModelId());
	for (auto entityId : entitySet)
	{
		BPEntity entity(entityId,*BPProject::getActiveProject());
		BPDataKey key = BPDataUtil::getDataKeyOnEntity(entity);
		if (key.isValid())
			vctKey.push_back(key);
	}

	//创建组件原型
	BPComponentPrototypePtr ptrPrototype = BPComponentPrototype::create();
	if (!BPComponentUtil::formComponentPrototype(vctKey, pProject, L"组件Demo", ptrPrototype))
		return;
	//创建组件实例
	BPComponentElementPtr ptrElement1 = BPComponentElement::create();
	bool bStatus2 = BPComponentUtil::formComponentElement(ptrPrototype, ptrElement1, pProject, GeTransform::createIdentityMatrix());
	BPComponentElementPtr ptrElement2 = BPComponentElement::create();
	BPComponentUtil::formComponentElement(ptrPrototype, ptrElement2, pProject, GeTransform::create(3000, 0, 0));
}


void createModelLine()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	BPModelBaseP pModel = pProject->getActiveModel();
	//创建线对象
	BPModelLine line(GePoint3d::create(3000, 3000, 0), GePoint3d::create(4000, 4000, 0));
	//将对象保存到model
	::p3d::P3DStatus status = line.addToProject(*pProject, pModel->getModelId());
	//改变线属性
	BPEntityArray entityArray;
	BPEntityUtil::getEntitiesOfModel(entityArray, *pProject, pModel->getModelId());
	BPEntityPtr entityptr = entityArray.getByIndex(0);
	//改变线型 0―P_Continuous、1-P_Dot、2-P_Dash1、3-P_Dash2、4-P_Dash_Dot1、5-P_Dash3、6-P_Dash_Double_dot、7-P_Dash_Dot2
	entityptr->setLineStyle(1);
	//改变线宽 线宽索引值0-23对应0.00mm-2.11mm，与CAD线宽相同
	entityptr->setWeight(23);
	//改变颜色 使用BPColorUtil，用于RGB颜色与索引色转换
	UInt32 color = BPColorUtil::getEntityColor(RGB(255, 0, 0), *pProject, true);
	entityptr->setColor(color);
	entityptr->replaceInModel(entityptr.get(), false);
}

void createModelHatch()
{	
	//获取所有填充类型
	BIMBase::Data::BPHatchPatternManager::getInstance().loadPatterns();
	pmap<PString, pvector<BPPat>> mapPat = BIMBase::Data::BPHatchPatternManager::getInstance().getPatterns();
	if (mapPat.size() == 0)
		return;
	//创建填充区域
	pvector<GePoint3d> vctPoints;
	vctPoints.push_back(GePoint3d::create(0, 0, 0));
	vctPoints.push_back(GePoint3d::create(3000, 0, 0));
	vctPoints.push_back(GePoint3d::create(3000, 3000, 0));
	vctPoints.push_back(GePoint3d::create(0, 3000, 0));
	GeCurveArrayPtr ptrCurveArray = GeCurveArray::createLinestringArray(vctPoints, GeCurveArray::BOUNDARY_TYPE_Outer);
	GeCurveArrayPtr ptrCurveRegion = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_ParityRegion);
	ptrCurveRegion->add(ptrCurveArray);
	//创建填充对象
	PString sError;	
	BPHatchPtr ptrHatch = BPHatch::create(mapPat.begin()->first, ptrCurveRegion.get(), sError);
	COLORREF colorDef = RGB(200, 0, 0);
	ptrHatch->setCustomSheetColor(colorDef);
	if (ptrHatch.isNull())
		return;
	//将对象绘制到相应model
	BIMBase::Core::BPProjectP pProject = BIMBase::Core::BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == NULL)
		return;
	BPModelBaseP pModel = pProject->getActiveModel();
	::p3d::P3DStatus status = ptrHatch->addToProject(*pProject, pModel->getModelId());
}

void IntersectionDemo()
{

	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	BPModelBaseP pModel = pProject->getActiveModel();
	BPGraphicsPtr ptrGraphic1 = pModel->createPhysicalGraphics();
	BPGraphicsPtr ptrGraphic2 = pModel->createPhysicalGraphics();
	BPGraphicsPtr ptrGraphic3 = pModel->createPhysicalGraphics();

	//创建填充区域
	pvector<GePoint3d> vctPoints;
	vctPoints.push_back(GePoint3d::create(0, 0, 0));
	vctPoints.push_back(GePoint3d::create(3000, 0, 0));
	vctPoints.push_back(GePoint3d::create(3000, 3000, 0));
	vctPoints.push_back(GePoint3d::create(0, 3000, 0));
	GeCurveArrayPtr ptrCurveArray = GeCurveArray::createLinestringArray(vctPoints, GeCurveArray::BOUNDARY_TYPE_Outer);
	ptrGraphic1->addGeCurveArray(*ptrCurveArray);

	pvector<GePoint3d> vctPoints2;
	vctPoints2.push_back(GePoint3d::create(2000, 0, -1000));
	vctPoints2.push_back(GePoint3d::create(2000, 4000, -1000));
	vctPoints2.push_back(GePoint3d::create(2000, 4000, 3000));
	vctPoints2.push_back(GePoint3d::create(2000, 0, 3000));
	GeCurveArrayPtr ptrCurveArray2 = GeCurveArray::createLinestringArray(vctPoints2, GeCurveArray::BOUNDARY_TYPE_Outer);
	ptrGraphic2->addGeCurveArray(*ptrCurveArray2);
	pvector<GeSegment3d> vecSeg;
	GeCurveFunction::calculateIntersectionSegmentsOfCurveRegions(*ptrCurveArray, *ptrCurveArray2, vecSeg);
	GeCurveArrayPtr ptrCurveArray3 = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_None);
	for (auto seg : vecSeg)
	{
		ptrCurveArray3->add(IGeCurveBase::createSegment(seg));
	}
	ptrGraphic3->addGeCurveArray(*ptrCurveArray3);

	ptrGraphic1->save();
	ptrGraphic2->save();
	ptrGraphic3->save();
}

void updateGraphics()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	BPModelBaseP pModel = pProject->getActiveModel();
	BPGraphicsPtr ptrGraphic1 = pModel->createPhysicalGraphics();
	BPGraphicsPtr ptrGraphic2 = pModel->createPhysicalGraphics();

	
	pvector<GePoint3d> vctPoints;
	vctPoints.push_back(GePoint3d::create(0, 0, 0));
	vctPoints.push_back(GePoint3d::create(3000, 0, 0));
	vctPoints.push_back(GePoint3d::create(3000, 3000, 0));
	vctPoints.push_back(GePoint3d::create(0, 3000, 0));
	GeCurveArrayPtr ptrCurveArray = GeCurveArray::createLinestringArray(vctPoints, GeCurveArray::BOUNDARY_TYPE_Outer);
	ptrGraphic1->addGeCurveArray(*ptrCurveArray);

	pvector<GePoint3d> vctPoints2;
	vctPoints2.push_back(GePoint3d::create(2000, 0, -1000));
	vctPoints2.push_back(GePoint3d::create(2000, 4000, -1000));
	vctPoints2.push_back(GePoint3d::create(2000, 4000, 3000));
	vctPoints2.push_back(GePoint3d::create(2000, 0, 3000));
	GeCurveArrayPtr ptrCurveArray2 = GeCurveArray::createLinestringArray(vctPoints2, GeCurveArray::BOUNDARY_TYPE_Outer);
	
	BPEntityId en = ptrGraphic1->save();
	BPEntity entity(en, *pProject);
	BPGraphicsUtils::copyPhysicalGraphics(*ptrGraphic2, *ptrGraphic1);
	ptrGraphic2->addGeCurveArray(*ptrCurveArray2);
	ptrGraphic2->updateEntityWithGraphics(&entity, false);
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun(_T("helloBIMBase"), &HelloBIMBase);
BPToolsManager::registerFun(_T("createGroupDemo"), &createGroup);
BPToolsManager::registerFun(_T("createComponentDemo"), &createComponent);
BPToolsManager::registerFun(_T("createModelLineDemo"), &createModelLine);
BPToolsManager::registerFun(_T("createModelHatchDemo"), &createModelHatch);
BPToolsManager::registerFun(_T("IntersectDemo"), &IntersectionDemo);
BPToolsManager::registerFun(_T("updateGraphics"), &updateGraphics);
BPToolsManager::registerFun(_T("crossFileToCopyDemo"), &crossFileToCopy);
AutoDoRegisterFunctionsEnd