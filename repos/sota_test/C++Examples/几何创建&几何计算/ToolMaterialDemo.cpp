#include "stdafx.h"
#include "ToolMaterialDemo.h"
#include "direct.h"
#include "ExampleUtilDemo.h"


ToolMaterialDemo::ToolMaterialDemo()
{
}


ToolMaterialDemo::~ToolMaterialDemo()
{
}

//图片路径
PString getPath()
{
	wchar_t* str = ExampleUtilDemo::getProjectPathCW();

	P3DFileName sFilePath(str);
	sFilePath.appendToDir(L"res\\test11.jpg");
	if (!P3DFileName::doesDirExist(sFilePath.c_str()))
		return _T("");

	return sFilePath.c_str();
}
PString getPath1()
{
	wchar_t* str = ExampleUtilDemo::getProjectPathCW();

	P3DFileName sFilePath(str);

	sFilePath.appendToDir(L"res\\test.jpg");
	if (!P3DFileName::doesDirExist(sFilePath.c_str()))
		return _T("");

	return sFilePath.c_str();
}


//实体创建材质
void ToolMaterialDemo::funMaterial()
{
	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return;
	BPProjectPtr ptrProject = pProjectManager->getMainProject();
	if (ptrProject.isNull())
		return;

	BPModelBaseP pModel = ptrProject->getActiveModel();
	if (pModel == nullptr)
		return;


	GeBoxInfo boxInfo(
		GePoint3d::createByZero(),
		GePoint3d::create(0, 0, 3000),
		GeVec3d::create(1, 0, 0),
		GeVec3d::create(0, 1, 0),
		3000, 3000,
		3000, 3000,
		true
	);
	IGeSolidBasePtr ptrSolid = IGeSolidBase::createGeBox(boxInfo);
	if (ptrSolid.isNull())
		return;

	BPGraphicsPtr ptrGrapic = pModel->createPhysicalGraphics();
	if (ptrGrapic.isNull())
		return;
	ptrGrapic->addGeSolidBase(*ptrSolid);
	ptrGrapic->finish();
	BPEntityId entityId = ptrGrapic->save();
	BPEntity entity(entityId, *pModel);


	BPMaterial mat;
	mat.mName = L"MatDemo";
	mat.mapFile = getPath();//L"D:\\PKPM.png";
	mat.isValid = true;
	//按尺寸贴图
	BPMaterial::setMaterialToEntity(entity, mat, true);

	//按比例贴图
	BPGraphicsPtr ptrGrapic2 = pModel->createPhysicalGraphics();
	if (ptrGrapic2.isNull())
		return;

	GeBoxInfo boxInfo1(
		GePoint3d::createByZero(),
		GePoint3d::create(0, 0, 3000),
		GeVec3d::create(1, 0, 0),
		GeVec3d::create(0, 1, 0),
		3000, 3000,
		3000, 3000,
		true
	);
	IGeSolidBasePtr ptrSolid2 = IGeSolidBase::createGeBox(boxInfo1);
	if (ptrSolid2.isNull())
		return;

	
	GeTransform trans = GeTransform::create(GePoint3d::create(5000, 0, 0));
	ptrSolid2->transform(trans);
	ptrGrapic2->addGeSolidBase(*ptrSolid2);
	ptrGrapic2->finish();
	BPEntityId entityId2 = ptrGrapic2->save();
	BPEntity entity2(entityId2, *pModel);

	BPMaterial mat2;
	mat2.mName = L"MatDemo2";
	mat2.mapFile = getPath1();//L"D:\\PKPM.png";
	mat2.isValid = true;
	mat2.mapUnit = BPMaterialMapUnit::mRelative;
	BPMaterial::setMaterialToEntity(entity2, mat2, true);
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("matDemo", ToolMaterialDemo::funMaterial);
AutoDoRegisterFunctionsEnd