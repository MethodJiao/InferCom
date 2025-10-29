#include "stdafx.h"
#include "ToolCreateTextTest.h"


ToolCreateTextTest::ToolCreateTextTest()
{
}


ToolCreateTextTest::~ToolCreateTextTest()
{
}

PBBuildingElementProxyPtr ToolCreateTextTest::createText(PString str, GePoint3d ptOri)
{
	PBBuildingElementProxyPtr pbProxy = PBBuildingElementProxy::create();
	if (pbProxy.IsNull())
		return pbProxy;

	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return pbProxy;
	BPProjectPtr projectPtr = pProjectManager->getMainProject();
	if (projectPtr.isNull())
		return pbProxy;

	BPModelPtr modelPtr = BPModel::getActiveModel();
	if (modelPtr.isNull())
		return pbProxy;

	BPFont font = BPFontUtil::getDefaultTrueTypeFont();

	p3d::GePoint2d pt = p3d::GePoint2d::create(1000, 1000);
	BPFont fontBig;
	BPTextPropertiesPtr textProperties = BPTextProperties::create(font, fontBig, pt, *modelPtr);
	if (textProperties.IsNull())
		return pbProxy;

	GeRotMatrix rotMatrix = GeRotMatrix::createIdentityMatrix();
	BPTextPtr ptrText = BPText::create(str.getWCharCP(), &ptOri, &rotMatrix, *textProperties);

	BPGraphicsPtr grapicPtr = modelPtr->createPhysicalGraphics();
	if (grapicPtr.isNull())
		return pbProxy;

	grapicPtr->addText(*ptrText);
	grapicPtr->save();

	pbProxy->AddPhysicalGraphics(*projectPtr, grapicPtr, PBBimCore::PBModelType::Physical);

	P3DStatus status = pbProxy->addToProject(*projectPtr);
	
	return pbProxy;
}

BPTextEntityPtr ToolCreateTextTest::createText2(PString str, GePoint3d ptOri)
{
	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return nullptr;
	BPProjectPtr projectPtr = pProjectManager->getMainProject();
	if (projectPtr.isNull())
		return nullptr;

	BPModelPtr modelPtr = BPModel::getActiveModel();
	if (modelPtr.isNull())
		return nullptr;

	PBModelInfoPtr modelInfoPtr = PBModelInfoManager::Get().GetModelById(modelPtr->getModelId());
	if (!modelInfoPtr.IsValid())
		return nullptr;

	BPTextStyle::constructDefaultTextStyle();
	BPTextEntityPtr text = BPTextEntity::create();
	if (text.isNull())
		return nullptr;
	text->setContent(str);
	text->setWidthFactor(1.0);
	text->setStyle(L"Œ¢»Ì—≈∫⁄");
	text->setPos(ptOri);
	text->setHeight(1.0);
	text->setStyle(L"sans-serif");
	text->setUseFixedHeight(false);

	text->addToProject(*projectPtr, modelInfoPtr->GetModelId());
	

	return text;
}

void run()
{
	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return;
	BPProjectPtr projectPtr = pProjectManager->getMainProject();
	if (projectPtr.isNull())
		return;


	BPModelPtr modelPtr = BPModel::getActiveModel();
	if (modelPtr.isNull())
		return;

	PString str = L" ÷≥µ’¢µ∂";
	BPComponentPrototypePtr comPrototype = BPComponentManager::getInstance(projectPtr).getPrototypeByName(str);
	if (comPrototype.isNull())
		return;

	CString strName = _T("±‡∫≈");
	BPComItemProp nameProperty;
	if (comPrototype->getPropItem(L"¿‡–Õ1", L"±‡∫≈√˚≥∆", nameProperty))
	{
		strName = nameProperty.psItemVal.at(0).c_str();
	}
	

	pvector<BPGraphicElementPtr> vctEle;
	comPrototype->getAllComponentElement(vctEle, *projectPtr);


	int nCount = 1;
	for (BPGraphicElementPtr ele : vctEle)
	{
		if (ele.isNull())
			continue;

		BPEntityPtr ee;
		(ele->getData(*projectPtr))->getElement(ee);
		//BPEntity ee(ele->getElementId(*projectPtr, modelPtr->getModelId()), *modelPtr);
		if(ee.isNull())
			continue;
		GeRange3d range = GeRange3d::createByNull();
		ee->getRange(range);
		GePoint3d pt = range.high;

		CString str;
		str.AppendFormat(_T("%d"), nCount);	
		str = strName + str;

		BPTextEntityPtr text = ToolCreateTextTest::createText2(str.GetString(), pt);//ToolCreateTextTest::createText(str.GetString(), pt);
		if (text.isNull())
			continue;

		BPDataKey keyText = text->getDataKey();

		KeyVct vctKey;
		vctKey.push_back(ele->getDataKey());
		vctKey.push_back(keyText);

		BPDataKey keyGroup = BPGroupUtil::group(vctKey, str);

		nCount++;
	}

}


AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("textTest", &run);
AutoDoRegisterFunctionsEnd