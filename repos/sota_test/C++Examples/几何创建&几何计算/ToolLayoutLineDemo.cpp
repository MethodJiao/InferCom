#include "stdafx.h"
#include "ToolLayoutLineDemo.h"


#define BIMBase_ATTACH_ENTITY_ID_Demo   2
#define BIMBase_ATTACH_ENTITY_PROP_Demo L"Demo123"

static DlgLayoutLineDemo* m_dlg;
ToolLayoutLineDemo::ToolLayoutLineDemo()
{
	m_nStep = 0;
	m_nInputWay = 0;
}


ToolLayoutLineDemo::~ToolLayoutLineDemo()
{
	if (m_dlg != nullptr)
		m_dlg->ShowWindow(SW_HIDE);
}


void ToolLayoutLineDemo::_onPostInstall()
{
	T_Super::_onPostInstall();
	PBBimModuleResourceOverride resOverride;
	if (m_dlg == nullptr)
	{
		m_dlg = new DlgLayoutLineDemo;
		m_dlg->Create(DlgLayoutLineDemo::IDD, AfxGetMainWnd());
		m_dlg->ShowWindow(SW_SHOWNOACTIVATE);
		m_nInputWay = m_dlg->m_nInputWay;
	}
	else
		m_dlg->ShowWindow(SW_SHOWNOACTIVATE);
	
	BPSnap::getInstance().enableLocate(false);
	BPSnap::getInstance().enableSnap(true);

	//追踪器设置
	showInputDlg(true);//关闭平台追踪器
	setToolPrompt(L"此处可输入交互字母:");
	__setTracer();

	PBMessageCenter::Send(PBBim_MESSAGE_ToolPrompt, Utf8String(L"请选择点").c_str());
}

void ToolLayoutLineDemo::_onRestartTool()
{
	ToolLayoutLineDemo* newTool = new ToolLayoutLineDemo();
	newTool->installTool();
}

bool ToolLayoutLineDemo::_onDataButton(BPBaseButtonEventCP ev)
{
	//获得用户的输入
	auto inpuntChar = getToolPrompt();
	
	BPViewportP pViewPort = BPViewManager::getInstance().getActivedViewport();
	if (NULL == pViewPort)
		return false;

	GePoint3d ptSel = *ev->getPoint();

	BPModelP pModel = pViewPort->getTargetModel();
	if (pModel == nullptr)
		return false;

	GePoint3d ptE = *ev->getPoint();

	switch (m_nInputWay)
	{
	case 0:
	{
		if (m_ptrGraphic.isNull())
			return false;

		m_ptrGraphic->save();

		//增加附加属性
		BPEntity entity = m_ptrGraphic->getEntityR();
		m_EntityIdSave = entity.getEntityId();
		int nDataSize = sizeof(BIMBase_ATTACH_ENTITY_PROP_Demo) + 10;
		::p3d::P3DStatus status = entity.setAttachData(BIMBase_ATTACH_ENTITY_ID_Demo, BIMBase_ATTACH_ENTITY_PROP_Demo, nDataSize, true);
		status = entity.replaceInModel(&entity);
	}
		break;
	case 1:
	{
		if (m_nStep == 0)
		{
			m_ptC = *ev->getPoint();
			m_nStep = 1;
		}
		else if (m_nStep == 1)
		{
			if (m_ptrGraphic.isNull())
				return false;

			m_ptrGraphic->save();
			m_nStep = 0;
		}
	}
		break;
	case 2:
	{
		m_vctPoint.push_back(*ev->getPoint());
		m_nStep = 1;
	}
		break;
	}

	return true;
}


bool ToolLayoutLineDemo::_onResetButton(BPBaseButtonEventCP)
{
	if (m_nInputWay == 2)
	{
		__createGraphic(false);
		m_ptrGraphic->save();
		__dynamicContinueDimension(false);
		m_vctPoint.clear();
	}
	if (m_EntityIdSave.isValid())
	{
		//获取附加的属性
		Int32 nLength = 0;
		
		BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
		if (pProject == nullptr)
			return false;

		BPModelBaseP pModel = pProject->getActiveModel();
		if (pModel == nullptr)
			return false;

		BPEntity entity(m_EntityIdSave, *pModel);
		entity.getAttachDataLength(nLength, BIMBase_ATTACH_ENTITY_ID_Demo);
		void* dataValue = malloc(nLength);
		entity.getAttachData(dataValue, nLength, BIMBase_ATTACH_ENTITY_ID_Demo);
		CString str = (LPCTSTR)dataValue;
		delete dataValue;
	}

	_exitTool();
	return true;
}

void ToolLayoutLineDemo::_onDynamicFrame(BPBaseButtonEventCP ev)
{
	if (m_dlg != nullptr)
		m_nInputWay = m_dlg->m_nInputWay;

	if (NULL == ev)
		return;

	if (m_nStep == 0 && m_nInputWay != 0)
	{
		return;
	}

	GePoint3d ptE = *ev->getPoint();
	__createGraphic(true, ptE);

	BPRedrawEntitys redrawElems;
	redrawElems.setDrawMode(BPDrawMode::enTempDraw);
	redrawElems.setDrawPurpose(BPDrawPurpose::enDynamics);
	redrawElems.setDynamicsViews(ev->getViewport());
	redrawElems.doRedraw(m_ptrGraphic->getEntityR());

	//动态标注
	if (m_nInputWay == 1)
	{
		BPGraphicsPtr ptrDem = __dynamicDimension(m_ptC, ptE);
		if (ptrDem.isNull())
			return;
		redrawElems.doRedraw(ptrDem->getEntityR());
	}
	else if (m_nInputWay == 2)
	{
		BPGraphicsPtr ptrDem = __dynamicContinueDimension(true, ptE);
		if (ptrDem.isNull())
			return;
		redrawElems.doRedraw(ptrDem->getEntityR());
	}
}

void ToolLayoutLineDemo::__createGraphic(bool beDynamic, GePoint3d endPoint)
{
	GeCurveArrayPtr ptrCurve = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_None);
	IGeCurveBasePtr ptrLine;

	pvector<GePoint3d> vctPoints;
	for (auto point : m_vctPoint)
	{
		vctPoints.push_back(point);
	}
	if (beDynamic)
		vctPoints.push_back(endPoint);
	
	switch (m_nInputWay)
	{
	case 0:
	{
		GeVec3d vec = GeVec3d::create(1, 1, 0);
		ptrLine = IGeCurveBase::createSegment(GeSegment3d::create(endPoint, endPoint + vec * 2000));
		if (ptrLine.isNull())
			return;
		ptrCurve->add(ptrLine);
	}
		break;
	case 1:
	{
		ptrLine = IGeCurveBase::createSegment(GeSegment3d::create(m_ptC, endPoint));
		if (ptrLine.isNull())
			return;
		ptrCurve->add(ptrLine);
	}
		break;
	case 2:
	{
		ptrCurve = GeCurveArray::createLinestringArray(vctPoints, GeCurveArray::BOUNDARY_TYPE_None);
	}
		break;
	default:
		break;
	}

	if (ptrCurve.isNull())
		return;

	BPModelP pModel = BPViewManager::getInstance().getActivedViewport()->getTargetModel();
	if (pModel == nullptr)
		return;
	m_ptrGraphic = pModel->createPhysicalGraphics();
	m_ptrGraphic->addGeCurveArray(*ptrCurve);
}

bool ToolLayoutLineDemo::_onModelMotion(BPBaseButtonEventCP ev)
{
	if (!getDynamicsStarted())
		_beginDynamics();
	return true;
}

void ToolLayoutLineDemo::__setTracer()
{
	m_tracerEv.m_u8RgId = "Tracer.SecDevCustom";
	m_tracerEv.m_bShowCoordBtn = true;
	m_tracerEv.m_bCoordBtnWork = true;
	m_tracerEv.m_Type2HR.insert(make_pair((UINT)TracerCusFunDemo::eStateType::TRACER_DEMO, make_pair(3, 6)));
	m_tracerEv.m_etype = (UINT)TracerCusFunDemo::eStateType::TRACER_DEMO;
	Tracer::TracerProxy::SetCurEvState(m_tracerEv);
	Tracer::TracerProxy::SetTracerType(Tracer::eTracerType::CUSTOM);
}


BPGraphicsPtr ToolLayoutLineDemo::__dynamicDimension(GePoint3d ptFirst, GePoint3d ptSecond)
{

	BIMBase::Core::BPProjectP pProject = BIMBase::Core::BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == NULL)
		return nullptr;

	//标注样式
	BIMBase::Core::BPDimensionStylePtr ptrDimensionStyle = BIMBase::Core::BPDimensionStyle::create(L"线性标注样式范例", *pProject);
	if (ptrDimensionStyle.isNull())
		return nullptr;

	ptrDimensionStyle->setDimtad(1);
	ptrDimensionStyle->setDimse1(true);
	ptrDimensionStyle->setDimse2(true);
	ptrDimensionStyle->setDimdec(0);
	ptrDimensionStyle->setDimrnd(0);
	ptrDimensionStyle->setDimscale(1);
	ptrDimensionStyle->setDimtxt(1000);
	ptrDimensionStyle->setDimasz(500);

	BIMBase::BPColorDef colorDef;
	colorDef.m_rgba.red = 125;
	colorDef.m_rgba.green = 125;
	colorDef.m_rgba.blue = 125;

	UInt32 colorInt = BPColorUtil::getEntityColor(colorDef, *pProject, true);
	ptrDimensionStyle->setDimclrd(colorInt);
	ptrDimensionStyle->replace(L"线性标注样式范例", pProject);

	//绘制选中对象的标注
	BIMBase::Data::BPPlacement dimPlace;
	GePoint3d xLine1Point = ptFirst;
	GePoint3d xLine2Point = ptSecond;
	GeVec3d dirVec = GeVec3d::createByStartEndNormalize(xLine1Point, xLine2Point);
	double ang = GeVec3d::create(1., 0., 0.).angleTo2D(dirVec);
	GeVec3d vOffset = dirVec;
	vOffset.rotate2D(PI/2);

	vOffset = vOffset * 2000;
	GePoint3d textPoint = xLine1Point;
	dirVec = dirVec * 0.5;
	textPoint = textPoint + vOffset;

	//根据计算信息构造直线标注,第6个参数传入不为空时，标注上显示传入的值
	BIMBase::Data::BPDimensionLinear linearDim(dimPlace, textPoint, xLine1Point, xLine2Point, textPoint, L"", ang);
	linearDim.setDimstyle(L"线性标注样式范例");

	BPGraphicsPtr  graphicsPtr = linearDim.createPhysicalGraphics(*pProject, pProject->getActiveModel()->getModelId(), true);
	return graphicsPtr;
}


BPGraphicsPtr ToolLayoutLineDemo::__dynamicContinueDimension(bool beDynamic, GePoint3d ptDynamic)
{
	if (m_vctPoint.size() <= 1)
		return nullptr;
	
	BIMBase::Core::BPProjectP pProject = BIMBase::Core::BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == NULL)
		return nullptr;

	//标注样式
	BIMBase::Core::BPDimensionStylePtr ptrDimensionStyle = BIMBase::Core::BPDimensionStyle::create(L"线性标注样式范例", *pProject);
	if (ptrDimensionStyle.isNull())
		return nullptr;

	ptrDimensionStyle->setDimtad(1);
	ptrDimensionStyle->setDimse1(true);
	ptrDimensionStyle->setDimse2(true);
	ptrDimensionStyle->setDimdec(0);
	ptrDimensionStyle->setDimrnd(0);
	ptrDimensionStyle->setDimscale(1);
	ptrDimensionStyle->setDimtxt(1000);
	ptrDimensionStyle->setDimasz(500);

	BIMBase::BPColorDef colorDef;
	colorDef.m_rgba.red = 125;
	colorDef.m_rgba.green = 125;
	colorDef.m_rgba.blue = 125;

	UInt32 colorInt = BPColorUtil::getEntityColor(colorDef, *pProject, true);
	ptrDimensionStyle->setDimclrd(colorInt);
	ptrDimensionStyle->replace(L"线性标注样式范例", pProject);

	//绘制选中对象的标注
	BIMBase::Data::BPPlacement dimPlace;

	vector<GePoint3d> vctPoint;
	vector<GePoint3d> vctTextPoint;
	vector<PString> vctText;
	double ang = 0;
	for (auto point : m_vctPoint)
	{
		vctPoint.push_back(point);			
	}	
	if (beDynamic)
	{
		vctPoint.push_back(ptDynamic);
	}
		
	for (int i = 1; i < vctPoint.size(); i++)
	{
		GeVec3d dirVec = GeVec3d::createByStartEndNormalize(vctPoint[i-1], vctPoint[i]);
		ang = GeVec3d::create(1., 0., 0.).angleTo2D(dirVec);
		GeVec3d vOffset = dirVec;
		vOffset.rotate2D(PI / 2);
		vOffset = vOffset * 2000;
		GePoint3d textPoint = vctPoint[i-1] + vOffset;
		vctTextPoint.push_back(textPoint);
		vctText.push_back(L"");
	}
	

	//根据计算信息构造直线标注,第6个参数传入不为空时，标注上显示传入的值
	BIMBase::Data::BPDimensionLinearContinuous linearDim(dimPlace, vctTextPoint, vctPoint, vctTextPoint.back(), vctText, 0);
	linearDim.setDimstyle(L"线性标注样式范例");
	
	if (beDynamic)
	{
		BPGraphicsPtr  graphicsPtr = linearDim.createPhysicalGraphics(*pProject, pProject->getActiveModel()->getModelId(), true);
		return graphicsPtr;
	}
	else
	{
		P3DStatus status = linearDim.addToProject(*pProject, pProject->getActiveModel()->getModelId());
		return nullptr;
	}
}

void ToolLayoutLineDemo::_onUserInput(const wchar_t* str, BPBaseButtonEventCP ev)
{
	if (str)
	{
		//用户自定义业务逻辑处理……

	}
	CString str1 = L"工具接收到按键输入:";
	CString str2(str);
	__sendMessage(str1 + str2);
}

// 提示信息
void ToolLayoutLineDemo::__sendMessage(const CString& msg)
{
	PBMessageCenter::Send(P3DApi::P3DApplication::JsonMessage(PBBim_MESSAGE_ToolPrompt, Utf8String(msg).c_str()));
}

void ToolLayoutLineDemo::_onTracerTabEvent(const Tracer::TracerInfo& info)
{
	__sendMessage(L"追踪器Tab按键响应事件");
}

void ToolLayoutLineDemo::_onTracerEditedChanged(const Tracer::TracerInfo& info, int nSelId)
{
	CString str1 = L"编辑框切换时事件,此时编辑框为:";
	CString str2;
	str2.Format(_T("%d"), nSelId);
	__sendMessage(str1 + str2);
}

void ToolLayoutLineDemo::_onTracerSureBtnClick(const Tracer::TracerInfo& info) 
{
	__sendMessage(L"追踪器点击确定按钮事件");
}

void ToolLayoutLineDemo::_onTracerCancelBtnClick() 
{
	__sendMessage(L"追踪器点击取消按钮事件");
}

void ToolLayoutLineDemo::_onTracerCoordBtnClick(const Tracer::TracerInfo& info) 
{
	__sendMessage(L"追踪器点击局部坐标按钮事件");
}

void ToolLayoutLineDemo::_onTracerEnterEdit()
{
	__sendMessage(L"追踪器进入编辑框事件");
}

void ToolLayoutLineDemo::_onTracerExitEdit() 
{
	__sendMessage(L"追踪器退出编辑框事件");
}


//对工具进行注册
BPTool* CreateLineTool()
{
	ToolLayoutLineDemo* tool = new ToolLayoutLineDemo();
	return tool;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("layoutLineDemo", &CreateLineTool);
AutoDoRegisterFunctionsEnd