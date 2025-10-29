#include "stdafx.h"
#include "BPCutModelManagerDemo.h"
#include "BPDrawingCenterDemo.h"
#include "BPDrawingCuttingDemo.h"
#include "BPDrawingInfoDemo.h"


using namespace DemoObject;
extern UInt32 newView;

static bool g_bFlag = true;
BPDrawingCenterDemo::BPDrawingCenterDemo()
{

}

BPDrawingCenterDemo::~BPDrawingCenterDemo()
{

}

void BPDrawingCenterDemo::doDrawingInfo()
{
	std::map<CString, PBModelInfoPtr> drawmodels = BPCutModelManagerDemo::Get().getModel();
	std::map<CString, PBModelInfoPtr>::iterator it = drawmodels.begin();
	for (; it != drawmodels.end(); it++)
	{
		PBModelInfoPtr model = it->second;
		if (model.isValid())
		{
			BPDrawingInfoDemo::Get().drawFrame(model);
			BPDrawingInfoDemo::Get().importFrame(model);
			BPDrawingInfoDemo::Get().drawBlock(model);
		}

	}
}

void BPDrawingCenterDemo::doDrawingCut()
{
	g_bFlag = true;
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;

	Params par = BPDrawingParasManagerDemo::Get().getParams();
	CString ModelName = par.cutModelName;
	BPDrawingParasManagerDemo::eDrawingview type = BPDrawingParasManagerDemo::Get().getDrawingview();
	PString sModelName = ModelName;

	GeRange3d range = GeRange3d::createByNull();

	BPDrawingCuttingDemo::Get().getAllModelRange(pProject, range);

	//剖切范围是所有对象的包围盒
	GePlane3d clipPlane;
	GeTransform sectionBox;
	GeTransform tm;
	//假设视图参数设置的事xy平面
	int types = 0;
	if (type == BPDrawingParasManagerDemo::eDrawingview::X_Y)
		types = 0;
	else if (type == BPDrawingParasManagerDemo::eDrawingview::Y_Z)
		types = 1;
	else if (type == BPDrawingParasManagerDemo::eDrawingview::X_Z)
		types = 2;
	BPDrawingCuttingDemo::Get().__createSectionBoxAndClipPlane(types, range, clipPlane, sectionBox, tm);
	//获取模型中所有的对象，把去剖切的和要符号化替代的分开
	BPViewportP pViewport = BPViewManager::getInstance().getViewport(0);
	if (pViewport == nullptr)
		return;

	BPModelP pModel = pViewport->getTargetModel();
	if (pModel == nullptr)
		return;
	PModelId mode = pModel->getModelId();
	//获取当前model上所有的图素
	BPEntityArray entityArray;
	BPEntityUtil::getEntitiesOfModel(entityArray, *pProject, pModel->getModelId());
	int  t = entityArray.getCount();
	if (entityArray.getCount() == 0)
	{
		AfxMessageBox(L"请布置路堤和排水沟");
		g_bFlag = false;
		return;
	}
		

	//确定剖切范围
	p3d::pvector<BPGraphicsPtr> pvecGraphics;//做剖切
	p3d::pvector<pair<BPDataKey, BPGraphicsPtr>>cutinstance;//做数据绑定
	//符号化的数据绑定
	p3d::pvector<pair<BPDataKey, BPGraphicsPtr>>cutinstancesym;//做数据绑定
	p3d::pvector<BPGraphicsPtr> pvecGraphicSymbol;//符号化后的gra结果
	GeTransform tran;
	tran.setByIdentityMatrix();
	bool bHasTarget = false;
	for (int i = 0; i < entityArray.getCount(); i++)
	{
		GeRange3d range3dew = GeRange3d::createByNull();
		BPEntityPtr ptrCurr = entityArray.getByIndex(i);
		if (!ptrCurr || !ptrCurr.isValid())
			continue;
		IBPObjectPtr ptrPbObj = BPObjectExtensionManager::getInstance().getBPObject(*ptrCurr);
		if (!ptrPbObj.isValid())
			continue;
		BPDemoGraphicElementDemoP pgraphicele = dynamic_cast<BPDemoGraphicElementDemoP>(ptrPbObj.get());
		if (pgraphicele == NULL)
			continue;

		bHasTarget = true;

		BPDataKey key = pgraphicele->getDataKey();
		BPGraphicsPtr gra = pgraphicele->createPhysicalGraphicsForDrawing(*pProject, pModel->getModelId());
		if (gra == nullptr)//说明需要剖切
		{


			BPGraphicsPtr ptrGraphic = BPEntityUtil::transformEntity(*ptrCurr, tran, false);
			if (ptrGraphic != nullptr)
			{
				pvecGraphics.push_back(ptrGraphic);
				cutinstance.push_back(make_pair(key, ptrGraphic));
			}


		}
		else//说明需要符号化
		{
			GeTransform  tr = GeTransform::createByProduct(tm, pgraphicele->getPlacement().toTransform());
			GePoint3d point;
			tr.getTranslation(point);
			tr = GeTransform::create(point);
			BPGraphicsUtils::transformPhysicalGraphics(*gra, tr);
			cutinstancesym.push_back(make_pair(key, gra));
		}


	}

	if (!bHasTarget)
	{
		AfxMessageBox(L"请布置路堤和排水沟");
		g_bFlag = false;
		return;
	}


	PBBimCore::PBModelInfoPtr modelInfoPtr = BPDrawingCuttingDemo::Get().getModelInfo(sModelName);
	if (!modelInfoPtr.isValid())
		return;
	CString sModelNames = sModelName.c_str();
	P3DModelUtil::DeleteElementsInModel(*pProject, modelInfoPtr->GetModelId(), true);

	if (cutinstance.size() != 0)
	{
		//单构件剖切
		//多构件剖切

		BPDrawingCuttingDemo::Get().cutting(modelInfoPtr, cutinstance, clipPlane, sectionBox, tm, pProject);
	}
	//符号化
	if (cutinstancesym.size() != 0)
	{
		PString modelname = sModelName;
		BPModelP ptrNewModel = pProject->loadModelById(modelInfoPtr->GetModelId());
		if (ptrNewModel != nullptr)
		{

			p3d::pvector<pair<BPDataKey, BPGraphicsPtr>>::iterator iteruse = cutinstancesym.begin();
			for (iteruse; iteruse != cutinstancesym.end(); iteruse++)
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

			GeRange3d ran = BPCutModelManagerDemo::Get().getModelRange(modelInfoPtr);
			BPViewportP pViewport = BPViewManager::getInstance().getViewport(newView);
			if (pViewport == nullptr)
				return;
			pViewport->lookAtVolume(ran);
			//创建的新model在view中显示
			BPViewManager::getInstance().displayModelOnViewPort(ptrNewModel->getModelId(), newView);
			BPViewManager::setAllow3DManipulations(newView, BPViewManager::BPRotateAxisOption::enRotateNone);
			AfxMessageBox(L"已储存");
		}
	}
	BPCutModelManagerDemo::Get().addModel(sModelNames, modelInfoPtr);

}

void BPDrawingCenterDemo::doDrawingDimension()
{
	std::map<CString, PBModelInfoPtr> drawmodels = BPCutModelManagerDemo::Get().getModel();
	std::map<CString, PBModelInfoPtr>::iterator it = drawmodels.begin();
	for (; it != drawmodels.end(); it++)
	{
		PBModelInfoPtr model = it->second;
		if (model.isValid())
			BPDrawingInfoDemo::Get().drawDimension(model);
	}
}

void BPDrawingCenterDemo::doDrawingTable()
{
	std::map<CString, PBModelInfoPtr> drawmodels = BPCutModelManagerDemo::Get().getModel();
	std::map<CString, PBModelInfoPtr>::iterator it = drawmodels.begin();
	for (; it != drawmodels.end(); it++)
	{
		CString name = it->first;
		PBModelInfoPtr model = it->second;
		if (model.isValid())
			BPDrawingInfoDemo::Get().drawTable(model);
	}
}



void BPDrawingCenterDemo::doDrawingLayout()
{
	std::map<CString, PBModelInfoPtr> drawmodels = BPCutModelManagerDemo::Get().getModel();
	BPDrawingInfoDemo::Get().layoutPic(drawmodels);
}

void BPDrawingCenterDemo::postProcessing()
{
	BPProjectP pProject = BPProject::getActiveProject();
	if (!pProject)
		return;
	BPModelPtr ptrModel = BPModel::getActiveModel();
	if (!ptrModel)
		return;
	BPGraphicsPtr ptrGraphic = ptrModel->createPhysicalGraphics();
	if (!ptrGraphic)
		return;
	//获取当前Frame的范围信息 最大的Frame就是工程图外框
	BPEntityArray entityArray;
	BPEntityUtil::getEntitiesOfModel(entityArray, *pProject, ptrModel->getModelId());
	if (entityArray.getCount() == 0)
		return;
	int nMaxRangeIndex = 0;
	GeRange3d maxRange = GeRange3d::createByNull();
	for (int i = 0; i < entityArray.getCount(); i++)
	{
		GeRange3d range3dew = GeRange3d::createByNull();
		BPEntityPtr ptrCurr = entityArray.getByIndex(i);
		if (!ptrCurr || !ptrCurr.isValid())
			continue;
		ptrCurr->getRange(range3dew);
		if (range3dew.high.y >= maxRange.high.y)
		{
			maxRange = range3dew;
			nMaxRangeIndex = i;
		}
	}
	//获取Frame右上角点坐标
	GePoint3d ptFrameRightUp = GeVec3d::createByZero();
	ptFrameRightUp = maxRange.high;
	//设置图例范围
	GePoint3d ptExampleRightUp = ptFrameRightUp - GeVec3d::create(14000, 4000, 0);
	GePoint3d ptExampleLeftDown = ptExampleRightUp - GeVec3d::create(10000, 2000, 0);

	//自定义图元绘制：直线
	GeCurveArrayPtr ptrCurveLine = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_None);
	IGeCurveBasePtr ptrLine;
	GePoint3d ptStartLine = GePoint3d::create(ptExampleLeftDown.x, ptExampleRightUp.y, 0);
	GeVec3d vecX = GeVec3d::create(1, 0, 0);
	ptrLine = IGeCurveBase::createSegment(GeSegment3d::create(ptStartLine, ptStartLine + vecX * 10000));
	if (ptrLine.isNull())
		return;
	ptrCurveLine->add(ptrLine);

	//自定义图元绘制：圆弧
	GeCurveArrayPtr ptrCurveArc = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_None);
	GePoint3d ptStartArc = GePoint3d::create(ptExampleLeftDown.x, ptExampleRightUp.y - 1500, 0);
	GePoint3d ptMidArc = GePoint3d::create(ptStartArc.x + 5000, ptStartArc.y - 1000, 0);
	GePoint3d ptEndArc = GePoint3d::create(ptStartArc.x + 10000, ptStartArc.y, 0);
	GeEllipse3d ell = GeEllipse3d::createByPointsOnEllipse(ptStartArc, ptMidArc, ptEndArc);
	IGeCurveBasePtr ptrArc = IGeCurveBase::createEllipse(ell);
	if (ptrArc.isNull())
		return;
	ptrCurveArc->add(ptrArc);

	//自定义图元绘制：多义线
	GeCurveArrayPtr ptrCurvePoly = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_None);
	for (int i = 0; i < 3; i++)
	{
		IGeCurveBasePtr ptrPolyLine;
		GePoint3d ptStartPolyLine = GePoint3d::create(ptExampleLeftDown.x + 2000 * 2 * i, ptExampleRightUp.y - 4000, 0);
		ptrPolyLine = IGeCurveBase::createSegment(GeSegment3d::create(ptStartPolyLine, ptStartPolyLine + vecX * 2000));
		if (ptrPolyLine.isNull())
			return;
		ptrCurvePoly->add(ptrPolyLine);
	}
	for (int i = 0; i < 2; i++)
	{
		GePoint3d ptStartPolyArc = GePoint3d::create(ptExampleLeftDown.x + 2000 * (2 * i + 1), ptExampleRightUp.y - 4000, 0);
		GePoint3d ptMidPolyArc = GePoint3d::create(ptStartPolyArc.x + 1000, ptStartPolyArc.y - 1000, 0);
		GePoint3d ptEndPolyArc = GePoint3d::create(ptStartPolyArc.x + 2000, ptStartPolyArc.y, 0);
		GeEllipse3d ell = GeEllipse3d::createByPointsOnEllipse(ptStartPolyArc, ptMidPolyArc, ptEndPolyArc);
		IGeCurveBasePtr ptrPolyArc = IGeCurveBase::createEllipse(ell);
		if (ptrPolyArc.isNull())
			return;
		ptrCurvePoly->add(ptrPolyArc);
	}

	Int32 lineStyle=0, arcLineStyle = 2, polyLineStyle = 0;
	//加载自定义线型(在acad.lin中添加自定义线型)
	PString sPathName = BPApplication::getInstance().getAppPath();
	PString sFileName = sPathName + L"PLATFORM\\linestyle\\acad.lin";
	BPLineStyleUtil::loadLinDefinition(sFileName.c_str(), L"DOTE", ptrModel.get(), 1.0);
	//指定自定义线型(以DOTE为例)
	const wchar_t* lineStyleName = L"DOTE";
	BPLineStyleMap lineStyleMap = BPLineStyleUtil::getLineStyleMapP(pProject);
	BPLineStyleInfo info;
	if (lineStyleMap.getLineStyleByName(lineStyleName, info));
	{
		lineStyle = BPLineStyleUtil::getNumberFromName(lineStyleName, *pProject);
	}
	//图例线型、线宽、颜色设置
	BPSymbology symbLine, symbArc, symbPolyLine;
	symbLine.style = lineStyle;
	symbArc.style = arcLineStyle;
	symbPolyLine.style = polyLineStyle;
	symbLine.weight = 1;
	symbArc.weight = 2;
	symbPolyLine.weight = 3;
	symbLine.color = 1;
	symbArc.color = 2;
	symbPolyLine.color = 3;
	ptrGraphic->addGeCurveArray(*ptrCurveLine, symbLine);
	ptrGraphic->addGeCurveArray(*ptrCurveArc, symbArc);
	ptrGraphic->addGeCurveArray(*ptrCurvePoly, symbPolyLine);
	ptrGraphic->save();



	Utf8CP                        EnShxFontName;
	Utf8CP                        ShxBigFontName;
	PString                       EnTrueTypeFontName;
	bool                          IsEnShxFont;
	BPFontType                    FontType;
	double                        FontHeight;
	double                        FontWidth;
	GePoint3d                     FontPosition;
	unsigned int                  FontColor;
	double                        FontItalic;
	P3DTextEntityJustification    FontJustification;
	::p3d::WCharCP                TextStyle;

	auto fun = [&](PString content)
	{

		//添加文字样式
		if (IsEnShxFont)
		{
			BPFont Font1 = BPFontUtil::findFont(EnShxFontName, FontType);
			BPFont Font2 = BPFontUtil::findFont(ShxBigFontName, FontType);
			BPTextStylePtr   _TextStyle = BPTextStyle::create(TextStyle, *pProject);
			P3DStatus status_1 = _TextStyle->setProperty(P3DTextStyleProperty::TextStyle_Font, Font1);
			P3DStatus status_2 = _TextStyle->setProperty(P3DTextStyleProperty::TextStyle_ShxBigFont, Font2);
			bool status_3 = _TextStyle->addToProject();
		}
		else
		{
			BPFont Font1 = BPFontUtil::findFont(EnTrueTypeFontName, FontType);
			BPTextStylePtr   _TextStyle = BPTextStyle::create(TextStyle, *pProject);
			P3DStatus status_1 = _TextStyle->setProperty(P3DTextStyleProperty::TextStyle_Font, Font1);
			bool status_2 = _TextStyle->addToProject();
		}

		//创建一个文字
		BPTextEntityPtr ptrText = BPTextEntity::create();
		if (ptrText.isNull())
			return;

		//文字属性设置
		ptrText->setPos(FontPosition);
		ptrText->setHeight(FontHeight);
		ptrText->setWidthFactor(FontWidth);
		ptrText->setUseFixedHeight(false);
		ptrText->setStyle(TextStyle);
		ptrText->setLineColor(FontColor);
		ptrText->setJustification(FontJustification);
		ptrText->setItalics(FontItalic);

		//单行文字内容设置
		ptrText->setContent(content);
		ptrText->addToProject(*pProject, ptrModel->getModelId());
	};


	FontWidth = 500;
	FontHeight = 1000;
	FontItalic = 0.0;
	FontJustification = P3DTextEntityJustification::enLeftTop;



	TextStyle = L"字体样式1";
	FontColor = 0;
	FontPosition = GePoint3d::create(ptExampleLeftDown.x - 4500, ptExampleRightUp.y + 500, 0);
	FontType = BPFontType::enShx;
	EnShxFontName = "txt";
	ShxBigFontName = "gbcbig";
	IsEnShxFont = true;
	fun(L"直线单行文字");

	TextStyle = L"字体样式2";
	FontColor = 1;
	FontPosition = GePoint3d::create(ptExampleLeftDown.x - 3000, ptExampleRightUp.y - 1000, 0);
	EnTrueTypeFontName = L"楷体";
	FontType = BPFontType::enTrueType;
	IsEnShxFont = false;;
	fun(L"圆弧");

	TextStyle = L"字体样式3";
	FontColor = 2;
	FontPosition = GePoint3d::create(ptExampleLeftDown.x - 4000, ptExampleRightUp.y - 3500, 0);
	EnTrueTypeFontName = L"微软雅黑";
	fun(L"多义线");

	AfxMessageBox(L"完成步骤2");

}

void cutDemo()
{
	BPDrawingCenterDemo draw;
	PString sModelName = L"CuttingModel";
	PString sModelName1 = L"SCuttingModel";
	Params pars;
	pars.cutModelName = sModelName.c_str();
	pars.strFrame = L"A0";
	CString legendPath = BPApplication::getInstance().getAppPath().c_str();
	pars.strLegend = legendPath;
	BPDrawingParasManagerDemo::Get().setParams(pars);
	BPDrawingParasManagerDemo::eDrawingview type = BPDrawingParasManagerDemo::eDrawingview::Y_Z;
	BPDrawingParasManagerDemo::Get().setDrawingview(type);
	draw.doDrawing();
	if (g_bFlag == false)
		return;

	pars.cutModelName = sModelName1.c_str();
	BPDrawingParasManagerDemo::Get().setParams(pars);
	type = BPDrawingParasManagerDemo::eDrawingview::X_Z;
	BPDrawingParasManagerDemo::Get().setDrawingview(type);
	draw.doDrawing();
	AfxMessageBox(L"完成步骤1");

}

//生成图纸
void drawmodelDemo()
{

	BPDrawingCenterDemo draw;
	draw.doDrawingLayout();
	draw.postProcessing();
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun(_T("cutDemo"), &cutDemo);
BPToolsManager::registerFun(_T("drawmodelDemo"), &drawmodelDemo);
AutoDoRegisterFunctionsEnd