#include "stdafx.h"
#include "EmbankmentDemo.h"

#define Property_Lenght                          "Length"
#define Property_TopWidth                           "TopWidth"
#define Property_GravelThickness                          "GravelThickness"
#define Property_PackingThickness                          "PackingThickness"
#define Property_Slop                          "Slop"


using namespace DemoObject;

::p3d::P3DStatus EmbankmentDemo::_initFromData(BIMBase::Core::BPDataCR instance)
{
	if (T_Super::_initFromData(instance) != P3DStatus::SUCCESS)
		return ERROR;

	BPValue value;
	P3DStatus status;

	status = instance.getValue(value, Property_Lenght);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setLength(value.getInteger());

	status = instance.getValue(value, Property_TopWidth);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setTopWidth(value.getInteger());

	status = instance.getValue(value, Property_GravelThickness);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setGravelThickness(value.getInteger());

	status = instance.getValue(value, Property_PackingThickness);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setPackingThickness(value.getInteger());

	status = instance.getValue(value, Property_Slop);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setSlop(value.getDouble());
	return SUCCESS;
}




::p3d::P3DStatus EmbankmentDemo::_copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const
{
	if (T_Super::_copyToData(instance, project) != P3DStatus::SUCCESS)
		return ERROR;

	P3DStatus status;
	status = instance.setValue(Property_Lenght, BPValue(this->getLength()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(Property_TopWidth, BPValue(this->getTopWidth()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(Property_GravelThickness, BPValue(this->getGravelThickness()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	status = instance.setValue(Property_PackingThickness, BPValue(this->getPackingThickness()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	status = instance.setValue(Property_Slop, BPValue(this->getSlop()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	return SUCCESS;
}



EmbankmentDemo::EmbankmentDemo()
{
	m_nTopWidth = 15000;
	m_nGravelThickness = 500;
	m_nPackingThickness = 2500;
	m_nLenght = 30000;
	m_dSlop = 1.5;
}

EmbankmentDemo::~EmbankmentDemo()
{

}

int EmbankmentDemo::getTopWidth() const
{
	return m_nTopWidth;
}

void EmbankmentDemo::setTopWidth(int nTopWidth)
{
	m_nTopWidth = nTopWidth;
}

int EmbankmentDemo::getLength() const
{
	return m_nLenght;
}

void EmbankmentDemo::setLength(int nLength)
{
	m_nLenght = nLength;
}

int EmbankmentDemo::getGravelThickness() const
{
	return m_nGravelThickness;
}

void EmbankmentDemo::setGravelThickness(int nGravelThickness)
{
	m_nGravelThickness = nGravelThickness;
}


int EmbankmentDemo::getPackingThickness() const
{
	return m_nPackingThickness;
}

void EmbankmentDemo::setPackingThickness(int nGravelThickness)
{
	m_nPackingThickness = nGravelThickness;
}

double EmbankmentDemo::getSlop() const
{
	return m_dSlop;
}

void EmbankmentDemo::setSlop(double dSlop)
{
	m_dSlop = dSlop;
}

BIMBase::Core::BPGraphicsPtr DemoObject::EmbankmentDemo::_createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool bIsDynamics)
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return nullptr;
	BPModelPtr ptrModel = project.getModelById(modelId);
	if (ptrModel.isNull())
		return nullptr;

	BPGraphicsPtr ptrGraphic = ptrModel->createPhysicalGraphics();
	if (ptrGraphic.isNull())
		return nullptr;

	double dDrainThickness = 100;
	double dDrainWidth = 2000;

	int nWidth = 0.5 * getTopWidth();
	int nLength = getLength();
	int nGravelThickness = getGravelThickness();
	int nPackingThickness = getPackingThickness();
	double dSlop = getSlop();
	double dXGravel = dSlop * nGravelThickness;
	double dXPacking = dSlop * nPackingThickness;
	double dXDrain = nWidth + dXGravel + dXPacking;
	double dXEdge = dXDrain - dSlop * dDrainThickness;


	GeVec3d veccc = GeVec3d::create(0, nLength, 0);

	Int32 nStyle = 0;
	BIMBase::BPColorDef colorRed(255, 0, 0);
	BIMBase::BPColorDef colorBlue(0, 0, 255);
	BIMBase::BPColorDef colorGray(120, 120, 120);

	UInt32 nDrainColor = BPColorUtil::getEntityColor(colorGray, *pProject, true);
	UInt32 nGravelColor = BPColorUtil::getEntityColor(colorRed, *pProject, true);
	UInt32 nPackingColor = BPColorUtil::getEntityColor(colorBlue, *pProject, true);


	BPSymbology sysDrain;
	sysDrain.color = nDrainColor;
	sysDrain.weight = 3;
	sysDrain.style = nStyle;

	BPSymbology sysGravel;
	sysGravel.color = nGravelColor;
	sysGravel.weight = 3;
	sysGravel.style = nStyle;

	BPSymbology sysPacking;
	sysPacking.color = nPackingColor;
	sysPacking.weight = 3;
	sysPacking.style = nStyle;


	//绘制填料层
	GeCurveArrayPtr ptrOutLinesPacking = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
	pvector<GePoint3d> ptsPacking;
	IGeCurveBasePtr ptrLinePacking;
	
	ptsPacking.push_back(GePoint3d::create(nWidth + dXGravel, 0, nPackingThickness));
	ptsPacking.push_back(GePoint3d::create(dXDrain, 0, 0));
	ptsPacking.push_back(GePoint3d::create(-dXDrain, 0, 0));
	ptsPacking.push_back(GePoint3d::create(-nWidth - dXGravel, 0, nPackingThickness));
	ptsPacking.push_back(GePoint3d::create(nWidth + dXGravel, 0, nPackingThickness));

	ptrLinePacking = IGeCurveBase::createLineString(ptsPacking);
	ptrOutLinesPacking->push_back(ptrLinePacking);
	GeExtrusionInfo extrDataPacking(ptrOutLinesPacking, veccc, true);
	IGeSolidBasePtr ptrExtrusionPacking = IGeSolidBase::createGeExtrusion(extrDataPacking);
	ptrGraphic->addGeSolidBase(*ptrExtrusionPacking, sysPacking);

	//绘制碎石层
	GeCurveArrayPtr ptrOutLinesGravel = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
	pvector<GePoint3d> ptsGravel;
	IGeCurveBasePtr ptrLineGravel;
	

	ptsGravel.push_back(GePoint3d::create(nWidth, 0, nGravelThickness + nPackingThickness));
	ptsGravel.push_back(GePoint3d::create(nWidth + dXGravel, 0, nPackingThickness));
	ptsGravel.push_back(GePoint3d::create(-nWidth - dXGravel, 0, nPackingThickness));
	ptsGravel.push_back(GePoint3d::create(-nWidth, 0, nGravelThickness + nPackingThickness));
	ptsGravel.push_back(GePoint3d::create(nWidth, 0, nGravelThickness + nPackingThickness));


	ptrLineGravel = IGeCurveBase::createLineString(ptsGravel);
	ptrOutLinesGravel->push_back(ptrLineGravel);
	GeExtrusionInfo extrDataGravel(ptrOutLinesGravel, veccc, true);
	IGeSolidBasePtr ptrExtrusionGravel = IGeSolidBase::createGeExtrusion(extrDataGravel);
	ptrGraphic->addGeSolidBase(*ptrExtrusionGravel, sysGravel);

	return ptrGraphic;
}



