#include "stdafx.h"
#include "ToolCreateTextDemo.h"
#include "BPDisplay/BPTextProperties.h"
#include "BPDisplay/BPText.h"
#include "BPDisplay/BPFontUtil.h"

ToolCreateTextDemo::ToolCreateTextDemo()
{
}

ToolCreateTextDemo::~ToolCreateTextDemo()
{
}

void ToolCreateTextDemo::CreateText()
{
	BPModelP pModel = BPViewManager::getInstance().getActivedViewport()->getTargetModel();

	GePoint2dCR pt2d = GePoint2d::create(1000, 1000);
	BPFontP pFoubig = NULL;
	BPTextPropertiesPtr ptrTextProp = BPTextProperties::create(BPFontUtil::getDefaultTrueTypeFont(), BPFont::BPFont(), pt2d, *pModel);

	//Utf8CP fontName = "gbcbig";  //设置shx文字
	//BPTextPropertiesPtr ptrTextProp = BPTextProperties::create(BPFontUtil::findFont(fontName, BPFontType::enShx), BPFont::BPFont(), pt2d, *pModel);
	if (ptrTextProp != nullptr)
	{
		ptrTextProp->setJustification(P3DTextEntityJustification::CenterTop);

	}
	GeTransform trans = GeTransform::createByOriginAndVectors(GePoint3d::create(0, 0, 0), GeVec3d::create(1, 0, 0), GeVec3d::create(0, 1, 0), GeVec3d::create(0, 0, 1));
	GeRotMatrix rotmat = GeRotMatrix::createByVectorAndRotationAngle(GeVec3d::create(0, 0, 1), 1);
	trans.getMatrix(rotmat);
	BPTextPtr ptrText1 = BPText::create(L"测试文字的对齐方式123", &GePoint3d::create(0, 3000, 0), &rotmat, *ptrTextProp);
	BPTextPtr ptrText2 = BPText::create(L"测试文字的对齐方式123456", &GePoint3d::create(0, 0, 0), &rotmat, *ptrTextProp);
	BPTextPtr ptrText3 = BPText::create(L"测试文字的对齐方式123456789", &GePoint3d::create(0, -3000, 0), &rotmat, *ptrTextProp);
	GePoint3d ptOrigin = GePoint3d::createByZero();
	ptrText1->setOriginFromUserOrigin(GePoint3d::create(0, 3000, 0));
	ptrText2->setOriginFromUserOrigin(GePoint3d::create(0, 0, 0));
	ptrText3->setOriginFromUserOrigin(GePoint3d::create(0, -3000, 0));

	if (ptrText1 != nullptr && ptrText2 != nullptr && ptrText3 != nullptr)
	{
		BPGraphicsPtr ptrGgrapic = pModel->createPhysicalGraphics();
		if (ptrGgrapic.isNull())
			return;
		ptrGgrapic->addText(*ptrText1);
		ptrGgrapic->addText(*ptrText2);
		ptrGgrapic->addText(*ptrText3);
		ptrGgrapic->save();
	}
}

void ToolCreateTextDemo::CreateTextEx()
{
	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return;
	BPProjectP pProject = pProjectManager->getMainProject();
	if (pProject == nullptr)
		return;
	BPModelBaseP pModel= pProjectManager->getActiveModel();
	if (pModel == nullptr)
		return;

	BPTextStyle::constructDefaultTextStyle();
	BPTextEntityPtr ptrText = BPTextEntity::create();
	if (ptrText.isNull())
		return;
	ptrText->setContent(_T("测试文字Ex"));
	ptrText->setWidthFactor(0.5);
	ptrText->setStyle(L"微软雅黑");
	ptrText->setPos(GePoint3d::create(0,0,0));
	ptrText->setHeight(3000);
	ptrText->setStyle(L"sans-serif");
	ptrText->setUseFixedHeight(false);

	ptrText->addToProject(*pProject, pModel->getModelId());
}

void ToolCreateTextDemo::CreateMText()
{
	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return;
	BPProjectP pProject = pProjectManager->getMainProject();
	if (pProject == nullptr)
		return;
	BPModelBaseP pModel = pProjectManager->getActiveModel();
	if (pModel == nullptr)
		return;
	//创建一个多行文字
	BPMTextEntityPtr ptrMtext = new BPMTextEntity();;
	BPFont font1 = BIMBase::Core::BPFontUtil::getDefaultTrueTypeFont();
	//多行文字之间的段落设置
	MTextParagraphPropertiesAppenderPtr appender = new MTextParagraphPropertiesAppender;
	appender->isFullJustified = false;
	appender->mtext_FirstLineIndent = 0;
	appender->mtext_HangingIndent = 0;
	appender->lineSpacingValue = 400;
	ptrMtext->appendTextPart(appender);
	//多行文字属性设置
	MTextRunPropertiesAppenderPtr runAppender = new MTextRunPropertiesAppender;
	runAppender->isItalic = false;
	runAppender->isOverlined = false;
	runAppender->font = font1;
	runAppender->overrideFontSize = true;
	runAppender->fontSize = GePoint2d::create(400, 600);
	ptrMtext->appendTextPart(runAppender);
	//多行文字内容设置
	MTextTextLineAppenderPtr textAppender1 = new MTextTextLineAppender;
	textAppender1->mtextLine = _T("测试多行文字行1");
	ptrMtext->appendTextPart(textAppender1);

	MTextLineBreakAppenderPtr ptrLineBreak = new MTextLineBreakAppender();
	ptrMtext->appendTextPart(ptrLineBreak);

	MTextTextLineAppenderPtr textAppender2 = new MTextTextLineAppender;
	textAppender2->mtextLine = _T("测试多行文字行2");	
	ptrMtext->appendTextPart(textAppender2);

	ptrMtext->addToProject(*pProject, pModel->getModelId());
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("textDemo", ToolCreateTextDemo::CreateText);
BPToolsManager::registerFun("textExDemo", ToolCreateTextDemo::CreateTextEx);
BPToolsManager::registerFun("textMDemo", ToolCreateTextDemo::CreateMText);
AutoDoRegisterFunctionsEnd