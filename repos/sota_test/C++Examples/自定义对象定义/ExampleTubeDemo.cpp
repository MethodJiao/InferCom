#include "stdafx.h"



using namespace DemoObject;


#define PROPERTY_PTSTART       "ptStart"
#define PROPERTY_PTEND         "ptEnd"
#define PROPERTY_TUBEDIAMETER  "tubeDiameter"
#define PROPERTY_TUBETHICKNESS "tubeThickness"
#define PROPERTY_POSITION_PTSTART "Position.ptStart" //结构体示例
#define PROPERTY_POSITION_PTEND  "Position.ptEnd" //结构体示例

ExampleTubeDemo::ExampleTubeDemo(GePoint3d ptStart,GePoint3d ptEnd, double dTubeDiameter, double dTubeThickness)
{
	m_ptStart = ptStart;
	m_ptEnd = ptEnd;
	m_dTubeDiameter = dTubeDiameter;
	m_dTubeThickness = dTubeThickness;
}

ExampleTubeDemo::ExampleTubeDemo()
{
	m_ptStart = GePoint3d::createByZero();
	m_ptEnd = GePoint3d::create(0, 1000, 0);
	m_dTubeDiameter = 100;
	m_dTubeThickness = 5;
}

void ExampleTubeDemo::setStartPoint(GePoint3d ptCenter)
{
	m_ptStart = ptCenter;
}

GePoint3d ExampleTubeDemo::getStartPoint() const
{
	return m_ptStart;
}

void DemoObject::ExampleTubeDemo::setEndPoint(GePoint3d ptEnd)
{
	m_ptEnd = ptEnd;
}

P3D_NAMESPACE_NAME::GePoint3d DemoObject::ExampleTubeDemo::getEndPoint() const
{
	return m_ptEnd;
}

void ExampleTubeDemo::setDiameter(double dTubeDiameter)
{
	if (dTubeDiameter < 0)
		m_dTubeDiameter = 100;
	m_dTubeDiameter = dTubeDiameter;
}

double ExampleTubeDemo::getDiameter() const
{
	return m_dTubeDiameter;
}

void ExampleTubeDemo::setThickness(double dTubeThickness)
{
	if (dTubeThickness < 0)
		m_dTubeThickness = 5;
	m_dTubeThickness = dTubeThickness;
}

double ExampleTubeDemo::getThickness() const
{
	return m_dTubeThickness;
}

::p3d::P3DStatus   ExampleTubeDemo::_initFromData(::BIMBase::Core::BPDataCR data)
{
	if (T_Super::_initFromData(data) != P3DStatus::SUCCESS)
		return ERROR;

	BPValue value;
	P3DStatus status;

	status = data.getValue(value, PROPERTY_TUBEDIAMETER);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setDiameter(value.getDouble());

	status = data.getValue(value, PROPERTY_TUBETHICKNESS);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setThickness(value.getDouble());

	status = data.getValue(value, PROPERTY_POSITION_PTSTART);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setStartPoint(value.getPoint3D());

	status = data.getValue(value, PROPERTY_POSITION_PTEND);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setEndPoint(value.getPoint3D());

	return SUCCESS;
}

::p3d::P3DStatus  ExampleTubeDemo::_copyToData(::BIMBase::Core::BPDataR data, ::BIMBase::Core::BPProjectR project) const
{
	if (T_Super::_copyToData(data, project) != P3DStatus::SUCCESS)
		return ERROR;

	P3DStatus status;
	status = data.setValue(PROPERTY_POSITION_PTSTART, BPValue(getStartPoint()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = data.setValue(PROPERTY_POSITION_PTEND, BPValue(getEndPoint()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = data.setValue(PROPERTY_TUBEDIAMETER, BPValue(getDiameter()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = data.setValue(PROPERTY_TUBETHICKNESS, BPValue(getThickness()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	return SUCCESS;
}

BIMBase::Core::BPGraphicsPtr   ExampleTubeDemo::_createPhysicalGraphics(::BIMBase::Core::BPProjectR project, ::BIMBase::PModelIdCR modelId, bool bIsDynamics)
{
	GeEllipse3d ellipseOuter = GeEllipse3d::createByPoints(
		GePoint3d::create(0, 0, 0),
		GePoint3d::create(getDiameter() / 2, 0, 0),
		GePoint3d::create(0, 0, getDiameter() / 2),
		0,
		360);
	IGeCurveBasePtr ptrOutterCuve = IGeCurveBase::createEllipse(ellipseOuter);
	if (ptrOutterCuve == nullptr)
		return nullptr;
	GeCurveArrayPtr ptrOutterCurveArray = GeCurveArray::create(GeCurveArray::BoundaryType::BOUNDARY_TYPE_Outer);
	ptrOutterCurveArray->add(ptrOutterCuve);

	GeEllipse3d ellipseInner = GeEllipse3d::createByPoints(
		GePoint3d::create(0, 0, 0),
		GePoint3d::create(getDiameter() / 2 - getThickness(), 0, 0),
		GePoint3d::create(0, 0, getDiameter() / 2 - getThickness()),
		0,
		360);
	IGeCurveBasePtr ptrInnerCuve = IGeCurveBase::createEllipse(ellipseInner);
	if (ptrInnerCuve == nullptr)
		return nullptr;
	GeCurveArrayPtr ptrInnerCurveArray = GeCurveArray::create(GeCurveArray::BoundaryType::BOUNDARY_TYPE_Inner);
	ptrInnerCurveArray->add(ptrInnerCuve);

	GeCurveArrayPtr ptrBaseCurve = GeCurveArray::create(GeCurveArray::BoundaryType::BOUNDARY_TYPE_ParityRegion);
	ptrBaseCurve->add(ptrOutterCurveArray);
	ptrBaseCurve->add(ptrInnerCurveArray);

	GeExtrusionInfo tubeInfo(
		ptrBaseCurve,
		GeVec3d::create(GePoint3d::create(0, m_ptStart.distance(m_ptEnd), 0)),
		true
	);
	IGeSolidBasePtr ptrTubeSolid = IGeSolidBase::createGeExtrusion(tubeInfo);
	if (ptrTubeSolid == nullptr)
		return nullptr;

	BPModelPtr ptrModel = project.loadModelById(modelId);
	if (ptrModel == nullptr)
		return nullptr;

	BPGraphicsPtr ptrGraphic = new BPGraphics(ptrModel);
	ptrGraphic->addGeSolidBase(*ptrTubeSolid);
	if (!ptrGraphic.isValid())
		return nullptr;

	///////////////////结合工程图图素-------------------
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
	BPHatchPtr ptrHatch = BPHatch::create(L"SOLID", ptrCurveRegion.get(), sError);
	if (ptrHatch.isNull())
		return ptrGraphic;
	COLORREF colorDef = RGB(200, 0, 0);
	ptrHatch->setCustomSheetColor(colorDef);
	
	BPGraphicsPtr ptrHatchGraphic = ptrHatch->createPhysicalGraphics(project, modelId, false);
	BPEntityArray arrayHatchGra = ptrHatchGraphic->getElementArray();
	for (int i = 0;i<arrayHatchGra.getCount();i++)
	{
		P3DStatus sta = ptrGraphic->setElement(*arrayHatchGra.getByIndex(i));
	}

	return ptrGraphic;
}

AutoDoRegisterFunctionsBegin
CString name = L"ExampleTubeDemo";
BPObjectExtensionManager::getInstance().registerBPObjectExtension(PBM_SCHEMA_Demo, static_cast<Utf8String>(name), new IndependentBridgeExtension());
AutoDoRegisterFunctionsEnd