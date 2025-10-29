#include "stdafx.h"
#include "BPDrawingInfoDemo.h"
#include "BPCutModelManagerDemo.h"
#include "BPDrawingCuttingDemo.h"
#include "BPDrawingParasManagerDemo.h"
#include "OdWriteEx/OdWriteEx.h"

#pragma comment(lib, "BPWriteReadEx.lib")

using namespace DemoObject;
extern UInt32 newView;
BPDrawingInfoDemo::BPDrawingInfoDemo()
{

}

BPDrawingInfoDemo::~BPDrawingInfoDemo()
{

}

BPDrawingInfoDemoR BPDrawingInfoDemo::Get()
{
	static BPDrawingInfoDemo single;
	return single;
}

BOOL readDwgFile(LPCTSTR fileName,
	PBModelInfoR model,
	GeTransformCR matrix = GeTransform::createIdentityMatrix(),
	COLORREF     backColor = RGB(0, 0, 0),
	void(*ptrMeterProgressFun)(int Pos) = nullptr,
	bool bIsMTMode = false,
	bool bEnablePartialLoading = false,
	bool bDisableSvcsOutput = false,
	bool bDsableRecompute = false,
	bool bDsableDump = false,
	bool bEnableAcisAudit = false
);

void BPDrawingInfoDemo::drawFrame(PBModelInfoPtr modelInfo)
{
	GeRange3d rangeResult = GeRange3d::createByNull();
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	BPModelP ptrNewModel = pProject->loadModelById(modelInfo->GetModelId());
	if (ptrNewModel == nullptr)
		return;
	//获取当前model上所有的图素
	BPEntityArray entityArray;
	BPEntityUtil::getEntitiesOfModel(entityArray, *pProject, modelInfo->GetModelId());

	if (entityArray.getCount() == 0)
		return;

	//确定剖切结果空间大小

	rangeResult = BPCutModelManagerDemo::Get().getModelRange(modelInfo);


	GePoint3d FrameCenter = GeVec3d::create((rangeResult.low.x + rangeResult.high.x) / 2, (rangeResult.low.y + rangeResult.high.y) / 2, 0);
	
	Params par = BPDrawingParasManagerDemo::Get().getParams();
	CString sName = par.strFrame;
	double xLength = 1189;
	if(sName == L"A1")
		xLength = 841;
	else if(sName == L"A2")
		xLength = 594;

	double dRangeX = fabs(rangeResult.low.x - rangeResult.high.x);
	double dRangeY = fabs(rangeResult.low.y - rangeResult.high.y);
	m_dFrameScale = dRangeX > dRangeY ? dRangeX * 1.3 / xLength : dRangeY * 1.3 / xLength;
	double scal = m_dFrameScale;
	//double xLength = (rangeResult.low.x + rangeResult.high.x) / 2;

	GePoint3d RightUp = FrameCenter + GeVec3d::create(xLength * scal / 2, 0.7074 * xLength * scal / 2, 0);
	GePoint3d LeftUp = FrameCenter + GeVec3d::create(-xLength * scal / 2, 0.7074 * xLength * scal / 2, 0);
	GePoint3d RightDown = FrameCenter + GeVec3d::create(xLength * scal / 2, -0.7074 * xLength * scal / 2, 0);
	GePoint3d LeftDown = FrameCenter + GeVec3d::create(-xLength * scal / 2, 0.7074 * -xLength * scal / 2, 0);
	//外层大框
	pvector<GePoint3d> vcPts;
	vcPts.push_back(RightUp);
	vcPts.push_back(LeftUp);
	vcPts.push_back(LeftDown);
	vcPts.push_back(RightDown);
	vcPts.push_back(RightUp);
	IGeCurveBasePtr Line = IGeCurveBase::createLineString(vcPts);
	if (Line.isNull())
		return;
	BPSymbology symbl = BPGraphics::getDefaultSymbology();
	symbl.style = BIMBase::STYLE_BYLAYER;
	symbl.weight = 0;
	symbl.color = BIMBase::COLOR_BYLAYER;
	BPGraphicsPtr graphicsaa = ptrNewModel->createPhysicalGraphics();
	graphicsaa->addGeCurve(*Line, symbl);
	graphicsaa->save();

	//里层小框
	GePoint3d RightUpa = RightUp;
	GePoint3d LeftUpa = LeftUp;
	GePoint3d RightDowna = RightDown;
	GePoint3d LeftDowna = LeftDown;
	if (xLength == 1189)
	{
		RightUpa = RightUp + GeVec3d::create(-20 * scal, -10 * scal, 0);
		LeftUpa = LeftUp + GeVec3d::create(25 * scal, -10 * scal, 0);
		RightDowna = RightDown + GeVec3d::create(-20 * scal, 10 * scal, 0);
		LeftDowna = LeftDown + GeVec3d::create(25 * scal, 10 * scal, 0);
	}
	//外层大框
	pvector<GePoint3d> vcPtsa;
	vcPtsa.push_back(RightUpa);
	vcPtsa.push_back(LeftUpa);
	vcPtsa.push_back(LeftDowna);
	vcPtsa.push_back(RightDowna);
	vcPtsa.push_back(RightUpa);
	IGeCurveBasePtr Linea = IGeCurveBase::createLineString(vcPtsa);
	if (Linea.isNull())
		return;
	BPSymbology symbla = BPGraphics::getDefaultSymbology();
	symbla.style = BIMBase::STYLE_BYLAYER;
	symbla.weight = 0;
	symbla.color = BIMBase::COLOR_BYLAYER;
	BPGraphicsPtr graphicsbb = ptrNewModel->createPhysicalGraphics();
	graphicsbb->addGeCurve(*Linea, symbla);
	graphicsbb->save();
	m_ptFrameInnerRD = RightDowna;
	m_ptFrameInnerRU = RightUpa;
}

void BPDrawingInfoDemo::importFrame(PBModelInfoPtr modelInfo) {

	Params par = BPDrawingParasManagerDemo::Get().getParams();
	CString sName = par.strLegend;

	P3DFileName sFilePath(sName);
	sFilePath.appendToDir(L"PKPM(2010版本).dwg");
	

	GeRange3d rangeResult = GeRange3d::createByNull();

	rangeResult = BPCutModelManagerDemo::Get().getModelRange(modelInfo);
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	BPModelP ptrNewModel = pProject->loadModelById(modelInfo->GetModelId());
	if (ptrNewModel == nullptr)
		return;


	void(*ptrMeterProgressFun)(int nPos) = [](int nPos)
	{
		BIMBase::FrameWork::ProgressCtrlTypeInfo info;
		info.b_DisplayPercent = false;
	};


	BPViewportP pViewPort = BPViewManager::getInstance().getActivedViewport();
	if (NULL == pViewPort)
	{
		return;
	}

	double xLength = 65688;

	double dRange = m_ptFrameInnerRU.y - m_ptFrameInnerRD.y;
	double scal = dRange / xLength;
	//缩放有问题，临时处理
	if (scal < 1)
		return;

	GeTransform tran;
	tran.setByOriginAndVectors(m_ptFrameInnerRD, GeVec3d::create(scal, 0, 0), GeVec3d::create(0, scal, 0), GeVec3d::create(0, 0, scal));


	BOOL bTemp = readDwgFile(sFilePath.c_str(), *modelInfo, tran, RGB(255, 255, 25), ptrMeterProgressFun);

}//画标注
void BPDrawingInfoDemo::drawDimension(PBModelInfoPtr modelInfo)
{
	BPProjectPtr project = BPProject::getActiveProject();
	if (!project.isValid())
		return;
	PString name = modelInfo->GetDisplayedName();
	if (name == L"Displaymodel")
		return;
	//获取当前model上所有的图素
	BPEntityArray entityArray;
	BPEntityUtil::getEntitiesOfModel(entityArray, *project, modelInfo->GetModelId());

	if (entityArray.getCount() == 0)
		return ;

	//确定剖切范围
	for (int i = 0; i < entityArray.getCount(); i++)
	{
		GeRange3d range3dew = GeRange3d::createByNull();
		BPEntityPtr ptrCurr = entityArray.getByIndex(i);
		if (!ptrCurr || !ptrCurr.isValid())
			continue;
		PString className;
		ptrCurr->getClassName(className);
		if (className == L"PBDimensionLinear")
			return;
		
	}
	
	//创建标注style
	BPDimensionStylePtr pStyle = BPDimensionStyle::create(L"DimensionStyle", *project);
	if (!pStyle.isValid())
		return;
	pStyle->addToProject();
	pStyle->setDimtad(1);//控制文字位置
	pStyle->setDimse1(true);//尺寸界限是否显示
	pStyle->setDimse2(true);//

	pStyle->setDimdec(0);
	pStyle->setDimrnd(0);

	double scale = 1;
	pStyle->setDimscale(scale);
	double txtH = 500;
	pStyle->setDimtxt(txtH);
	double dGap = 20;
	double dDist1 = 200;
	double dDist2 = 200;
	double dXieLenTmp = 300;
	double dXieLen = 300;
	double dDist3 = 0.1;
	int16_t iArrow = 3;
	pStyle->setDimgap(dGap);
	pStyle->setDimdle(dDist1);
	pStyle->setDimexe(dDist2);
	pStyle->setDimexo(dDist3);
	pStyle->setDimatfit(iArrow);
	pStyle->setDimasz(dXieLenTmp);
	pStyle->setDimtsz(dXieLen);

	pStyle->setDimtix(true);
	BPColorDef textClr;
	textClr.m_rgb.red = 255;
	textClr.m_rgb.green = 0;
	textClr.m_rgb.blue = 0;

	UInt32 txtcolor = BPColorUtil::getEntityColor(textClr, *project, true);
	pStyle->setDimclrt(txtcolor);
	BPColorDef dimClr;
	dimClr.m_rgb.red = 0;
	dimClr.m_rgb.green = 255;
	dimClr.m_rgb.blue = 0;

	UInt32 dimcolor2 = BPColorUtil::getEntityColor(dimClr, *project, true);
	pStyle->setDimclrd(dimcolor2);//尺寸线颜色
	pStyle->setDimclre(dimcolor2);
	pStyle->replace(pStyle->getName().c_str(), project.get());

	double dInterval = 900;
	GeRange3d range = BPCutModelManagerDemo::Get().getModelRange(modelInfo);
	//----------上面总标注-------------
	BPDimensionLinear pLineSingleDimUp;
	GePoint3d ptS = GePoint3d::createByZero();
	GePoint3d ptE = GePoint3d::createByZero();
	GePoint3d ptText = GePoint3d::createByZero();
	ptS = GePoint3d::create(range.low.x, range.high.y, 0);
	ptE = GePoint3d::create(range.high.x - 1, range.high.y, 0);
	ptText = GePoint3d::create(range.high.x, range.high.y + dInterval, 0);
	pLineSingleDimUp.setFirstXlinePt(ptS);

	pLineSingleDimUp.setSecondXlinePt(ptE);
	pLineSingleDimUp.setDimstyle(_T("DimensionStyle"));
	pLineSingleDimUp.setRotAngle(0);
	pLineSingleDimUp.setDefinedPoint(ptText);
	pLineSingleDimUp.addToProject(*project, modelInfo->GetModelId());
	
	//----------下面总标注-------------
	BPDimensionLinear pLineSingleDimDown;

	ptS = GePoint3d::create(range.low.x, range.low.y, 0);
	ptE = GePoint3d::create(range.high.x - 1, range.low.y, 0);
	ptText = GePoint3d::create(range.high.x, range.low.y - dInterval, 0);
	pLineSingleDimDown.setFirstXlinePt(ptS);

	pLineSingleDimDown.setSecondXlinePt(ptE);
	pLineSingleDimDown.setDimstyle(_T("DimensionStyle"));
	pLineSingleDimDown.setRotAngle(0);
	pLineSingleDimDown.setDefinedPoint(ptText);
	pLineSingleDimDown.addToProject(*project, modelInfo->GetModelId());

	//----------左面总标注-------------
	BPDimensionLinear pLineSingleDimLeft;

	ptS = GePoint3d::create(range.low.x, range.low.y, 0);
	ptE = GePoint3d::create(range.low.x, range.high.y, 0);
	ptText = GePoint3d::create(range.low.x - dInterval, range.high.y, 0);
	pLineSingleDimLeft.setFirstXlinePt(ptS);

	pLineSingleDimLeft.setSecondXlinePt(ptE);
	pLineSingleDimLeft.setDimstyle(_T("DimensionStyle"));
	pLineSingleDimLeft.setRotAngle(PI / 2);
	pLineSingleDimLeft.setDefinedPoint(ptText);
	pLineSingleDimLeft.addToProject(*project, modelInfo->GetModelId());

	//----------右面总标注-------------
	BPDimensionLinear pLineSingleDimRight;

	ptS = GePoint3d::create(range.high.x, range.low.y, 0);
	ptE = GePoint3d::create(range.high.x, range.high.y, 0);
	ptText = GePoint3d::create(range.high.x + dInterval, range.high.y, 0);
	pLineSingleDimRight.setFirstXlinePt(ptS);

	pLineSingleDimRight.setSecondXlinePt(ptE);
	pLineSingleDimRight.setDimstyle(_T("DimensionStyle"));
	pLineSingleDimRight.setRotAngle(-PI / 2);
	pLineSingleDimRight.setDefinedPoint(ptText);
	pLineSingleDimRight.addToProject(*project, modelInfo->GetModelId());
}


void BPDrawingInfoDemo::drawTable(PBModelInfoPtr modelInfo)
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;

	//获取当前model上所有的图素
	BPEntityArray entityArray;
	BPEntityUtil::getEntitiesOfModel(entityArray, *pProject, modelInfo->GetModelId());
	GeRange3d range = BPCutModelManagerDemo::Get().getModelRange(modelInfo);
	std::map<CString, BPDemoGraphicElementDemoP> tabledata;
	vector<BPEntityPtr> pgraphiceles;
	CString elementgoalblename = L"";
	int si = entityArray.getCount();
	int cc = 0;
	for (int i = 0; i < entityArray.getCount(); i++)
	{
		BPEntityPtr ptrCurr = entityArray.getByIndex(i);
		if (!ptrCurr || !ptrCurr.isValid())
			continue;

		
			PString className;
			ptrCurr->getClassName(className);
			if (className == L"EmbankmentDemo")
				cc++;

		
		IBPObjectPtr ptrPbObj = BPObjectExtensionManager::getInstance().getBPObject(*ptrCurr);
		if (!ptrPbObj.isValid())
			continue;
		//pbObjs.push_back(ptrPbObj);
		BPDemoGraphicElementDemoP pgraphicele = dynamic_cast<BPDemoGraphicElementDemoP>(ptrPbObj.get());
		if (pgraphicele != NULL)
		{
			pgraphiceles.push_back(ptrCurr);
			//记录一下有哪些对象的类名
			Utf8String sClassName = pgraphicele->getClassName();
			PString sname;
			P3DStringHelper::utf8ToWChar(sname, sClassName.c_str());
			CString sName = sname.c_str();
			if (elementgoalblename != sName)
			{
				elementgoalblename = sName;
				tabledata[elementgoalblename] = pgraphicele;
			}


		}

	}

	std::map<CString, int> elementcount;
	//找到每个类的对象的个数
	std::map<CString, BPDemoGraphicElementDemoP>::iterator it = tabledata.begin();
	for (; it != tabledata.end(); it++)
	{
		int count = 0;
		CString name = it->first;
		for (int i = 0; i < pgraphiceles.size(); i++)
		{
			BPEntityPtr pgraphicelement = pgraphiceles.at(i);
			if (pgraphicelement != NULL)
			{
				PString sname;
				pgraphicelement->getClassName(sname);
				CString sName = sname.c_str();
				if (name == sName)
					count++;
			}
		}
		elementcount[name] = count;
	}


	//创建表格,现在的内容就先只统计
	BPTableEntityP table = new BPTableEntity;
	table->setRowNum(tabledata.size() + 2);
	table->setColumnNum(2);
	table->setColWidths(10000);
	table->setRowHeight(2000);
	table->addCellForTable();

	BPTableEntityManager tablemanager = BPTableEntityManager(table);

	table->setTransform(p3d::GeTransform::create(0, range.high.y - 10 * m_dFrameScale, 0));
	p3d::P3DStatus status = table->addToProject(*pProject, modelInfo->GetModelId());
	auto fun = [&](PString sCur, bool bUnderlinedCur, bool bBold, UInt32 colorCur,
		double HangingIndentCur, int cellR, int cellC)//(文字内容 是否下划线 是否加粗 颜色 悬挂缩进量 所在行 所在列)
	{
		//创建一个文字
		BPMTextEntity pbmtext;
		BPFont font1 = BIMBase::Core::BPFontUtil::getDefaultTrueTypeFont();
		//多行文字之间的段落设置
		MTextParagraphPropertiesAppenderPtr ptrAppender = new MTextParagraphPropertiesAppender;
		ptrAppender->isFullJustified = false;
		ptrAppender->mtext_FirstLineIndent = 0;
		ptrAppender->mtext_HangingIndent = HangingIndentCur;
		pbmtext.appendTextPart(ptrAppender);
		//多行文字属性设置
		MTextRunPropertiesAppenderPtr ptrRunAppender = new MTextRunPropertiesAppender;
		ptrRunAppender->color = colorCur;
		ptrRunAppender->isBold = bBold;
		ptrRunAppender->isItalic = false;
		ptrRunAppender->isUnderlined = bUnderlinedCur;
		ptrRunAppender->isOverlined = false;
		ptrRunAppender->font = font1;
		ptrRunAppender->overrideFontSize = true;
		ptrRunAppender->fontSize = GePoint2d::create(400, 600);
		pbmtext.appendTextPart(ptrRunAppender);
		//多行文字内容设置
		MTextTextLineAppenderPtr ptrTextAppender = new MTextTextLineAppender;
		ptrTextAppender->mtextLine = sCur;
		pbmtext.appendTextPart(ptrTextAppender);
		tablemanager.setContextEntityForCell(&pbmtext, cellR, cellC);
	};

	//表头
	fun(L"属性表", false, true, 2, 0, 0, 1);
	fun(L"构件\n这里是多行文字", true, false, 0, 0, 1, 0);
	fun(L"统计个数", false, false, 0, 0, 1, 1);
	//合并第一行单元格
	BPTableCellEntity cell1, cell2;
	cell1.m_rowIndex = 0;
	cell1.m_colIndex = 0;
	cell2.m_rowIndex = 0;
	cell2.m_colIndex = 1;
	tablemanager.mergeCellsInTable(cell1, cell2);
	//表格内容
	std::map<CString, int>::iterator ele = elementcount.begin();
	int n = 0;
	for (; ele != elementcount.end(); ele++)
	{
		CString str, strL;
		str.Format(_T("%s"), ele->first);
		strL.Format(_T("%d"), ele->second/2);

		fun(str.GetString(), false, false, 0, 0, n + 2, 0);
		fun(strL.GetString(), false, false, 0, 0, n + 2, 1);
		n++;
	}
		
}

void BPDrawingInfoDemo::layoutPic(std::map<CString, PBModelInfoPtr> modelinfo)
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	std::map<CString, PBModelInfoPtr>::iterator it = modelinfo.begin();
	p3d::pvector<pair<GeRange3d, PBModelInfoPtr>> rangemodel;
	vector<GeRange3d> ranges;
	for (; it != modelinfo.end();it++)
	{
		CString name = it->first;
		if (name == L"Displaymodel")
			continue;
		PBModelInfoPtr model = it->second;
		if (!model.isValid())
			continue;
		//BPDrawingInfoDemo::Get().drawFrame(model);
		GeRange3d range = BPCutModelManagerDemo::Get().getModelRange(model);
		ranges.push_back(range);
		rangemodel.push_back(make_pair(range, model));
	}
	p3d::pvector<pair<GeRange3d, PBModelInfoPtr>>::iterator itmodel = rangemodel.begin();
	//把多个model里的东西放一起
	// 创建一个显示图纸的model
	PString sModelName = L"Displaymodel";
	PBBimCore::PBModelInfoPtr modelInfodPtr = BPDrawingCuttingDemo::Get().getModelInfo(sModelName);
	if (!modelInfodPtr.isValid())
		return;

	PModelId id = modelInfodPtr->GetModelId();
	P3DModelUtil::DeleteElementsInModel(*pProject, id, true);

	CString sModelNames = sModelName.c_str();
	GePoint3d targetori = GePoint3d::create(0, 0, 0);

	for (; itmodel != rangemodel.end(); itmodel++)
	{
		int bb = 0;
		GeRange3d range = itmodel->first;
		PBModelInfoPtr modelin = itmodel->second;
		//GePoint3d  targetori = GePoint3d::create(range.low.x, range.low.y, 0) + GePoint3d::create((range.high.x - range.low.x + 5000), range.low.y, 0);//第二次要放的点，左下角要放的位置
		GePoint3d ori = GePoint3d::create(range.low.x, range.low.y, 0);
		GePoint3d movept = targetori - ori;
		targetori = targetori + GePoint3d::create((range.high.x - range.low.x + 5000), range.low.y, 0);

		//获取当前model上所有的图素
		BPEntityArray entityArray;
		BPEntityUtil::getEntitiesOfModel(entityArray, *pProject, modelin->GetModelId());
		BPModelP ptrNewModel = pProject->loadModelById(modelInfodPtr->GetModelId());

		if (entityArray.getCount() == 0)
			continue;
		for (int i = 0; i < entityArray.getCount(); i++)
		{
			GeRange3d range3dew = GeRange3d::createByNull();
			BPEntityPtr ptrCurr = entityArray.getByIndex(i);
			if (!ptrCurr || !ptrCurr.isValid())
				continue;
			PString className;
			ptrCurr->getClassName(className);
			if (className == L"EmbankmentDemo")
				bb++;
			BPDataKey datakey = BPDataUtil::getDataKeyOnEntity(*ptrCurr);
			
			GeTransform tran;
			tran.setByIdentityMatrix();
			BPGraphicsPtr ptrGraphic = BPGraphics::getGraphicsFromEntity(*ptrCurr.get());
			if (!ptrGraphic.isValid())
				continue;
			ptrCurr->getRange(range3dew);
			ptrGraphic->setModel(ptrNewModel);
			GeTransform trans;
			trans.setByIdentityMatrix();
			
			trans.setByOriginAndVectors(movept,GeVec3d::create(1,0,0), GeVec3d::create(0, 1, 0), GeVec3d::create(0, 0, 1));
			ptrGraphic->setTransform(trans,true);
			BPEntityId entit = ptrGraphic->save();
			if (entit.isValid())
			{
				BPDataUtil::bindEntityToData(entit, datakey, pProject);
			}

		}
	}

	BPDrawingInfoDemo::Get().drawFrame(modelInfodPtr);
	BPDrawingInfoDemo::Get().drawTable(modelInfodPtr);
	BPDrawingInfoDemo::Get().importFrame(modelInfodPtr);
	BPDrawingInfoDemo::Get().drawBlock(modelInfodPtr);
	BPCutModelManagerDemo::Get().addModel(sModelNames, modelInfodPtr);
	vector<int> activeViewSet;
	BPViewManager::getInstance().getAllActiveViewports(activeViewSet);
	bool ismutiviews = activeViewSet.size() == 1 ? false : true;
	if (!ismutiviews)
	{
		BIMBase::BPUserInputManager::exeCommand("view_style_OPEN_NEW");
		newView = BPViewManager::getInstance().getActiveIndex();
	}

	//创建的新model在view中显示
	BPViewManager::getInstance().displayModelOnViewPort(modelInfodPtr->GetModelId(), newView);
	BPViewManager::setAllow3DManipulations(newView, BPViewManager::BPRotateAxisOption::enRotateNone);
}

void DemoObject::BPDrawingInfoDemo::drawBlock(PBModelInfoPtr modelInfo)
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;

	GeRange3d range = BPCutModelManagerDemo::Get().getModelRange(modelInfo);

	BIMBase::Core::BPTextStylePtr ptrTextStyle = BIMBase::Core::BPTextStyle::getActiveStyle(*pProject);
	if (ptrTextStyle.isNull())
		return;

	BPTextEntity text;
	text.setStyle(ptrTextStyle->getName());

	text.setContent(L"图块测试");
	text.setUseFixedHeight(true);
	text.setUseFixedWidFactor(true);
	text.setHeight(500);
	text.setWidthFactor(0.5);
	text.setRotation(0);
	text.setPos(GePoint3d{ 2000,2000,0 });

	BPGraphicsPtr ptrGraphicsText = text.createPhysicalGraphics(*pProject,modelInfo->GetModelId(), false);
	if (ptrGraphicsText.isNull())
		return;

	BPModelPtr ptrModel = pProject->getModelById(modelInfo->GetModelId());
	if (ptrModel.isNull())
		return;

	BPGraphicsPtr ptrGraphicsSeg = ptrModel->createPhysicalGraphics();
	if (ptrGraphicsSeg.isNull())
		return;

	GePoint3d pt = GePoint3d::create(100, 100, 100);
	IGeCurveBasePtr ptrCurve = IGeCurveBase::createLineString({ GePoint3d{ 0,0,0 }, GePoint3d{2000,2000,0} });
	if (ptrCurve.isNull())
		return;

	BPSymbology symb;
	symb.color = 2;
	symb.style = 0;
	symb.weight = 64;

	ptrGraphicsSeg->addGeCurve(*ptrCurve, symb);
	ptrGraphicsSeg->finish();
	BPSharedBlockDefEntity mBlock;
	BPEntityArray entityArray;
	BPEntityArrayR textArray = ptrGraphicsText->getElementArray();
	entityArray.insert(*(textArray.getByIndex(0)));
	entityArray.insert(ptrGraphicsSeg->getEntityR());
	mBlock.setName(L"BlockDemo");
	mBlock.setEntityArray(entityArray);
	mBlock.addToProject(*pProject, modelInfo->GetModelId());

	BPSharedBlockEntity mBlockEntity;
	mBlockEntity.setName(L"BlockDemo");
	mBlockEntity.setOrigin(GePoint3d::create(range.low.x + 150 * m_dFrameScale, range.high.y - 150 * m_dFrameScale, 0));
	mBlockEntity.addToProject(*pProject, modelInfo->GetModelId());

	BPSharedBlockEntity mBlockEntity1;
	mBlockEntity1.setName(L"BlockDemo");
	mBlockEntity1.setOrigin(GePoint3d::create(range.low.x + 150 * m_dFrameScale, range.high.y - 150 * m_dFrameScale-1200, 0));
	mBlockEntity1.addToProject(*pProject, modelInfo->GetModelId());
}
