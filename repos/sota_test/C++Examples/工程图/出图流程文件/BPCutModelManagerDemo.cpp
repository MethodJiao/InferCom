#include "stdafx.h"
#include"BPCutModelManagerDemo.h"

using namespace DemoObject;


void BPCutModelManagerDemo::addModel(CString modelName, PBModelInfoPtr models)
{
	m_Namemodels[modelName] = models;
}

BPCutModelManagerDemoR BPCutModelManagerDemo::Get()
{
	static BPCutModelManagerDemo single;
	return single;
}

GeRange3d DemoObject::BPCutModelManagerDemo::getModelRange(PBModelInfoPtr outModel)
{
	GeRange3d rangeResult = GeRange3d::createByNull();
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return rangeResult;
	BPModelP ptrNewModel = pProject->loadModelById(outModel->GetModelId());
	if (ptrNewModel == nullptr)
		return rangeResult;
	
	//获取当前model上所有的图素
	BPEntityArray entityArray;
	BPEntityUtil::getEntitiesOfModel(entityArray, *pProject, ptrNewModel->getModelId());

	if (entityArray.getCount() == 0)
		return rangeResult;

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
		rangeResult.extendRange(range3dew);
	}
	return rangeResult;

 }


void DemoObject::BPCutModelManagerDemo::deleteModel(PBModelInfoPtr  modelDel) {

	for (auto it = m_models.begin(); it != m_models.end();)
	{
		if (modelDel == *it) {
			it = m_models.erase(it);
		}
		else {
			++it;
		}
	}

}



