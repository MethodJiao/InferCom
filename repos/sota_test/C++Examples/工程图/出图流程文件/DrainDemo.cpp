#include "stdafx.h"
#include "DrainDemo.h"

#define Property_Lenght                          "Length"
#define Property_Width                           "Width"
#define Property_Thickness                          "Thickness"
#define Property_Depth                          "Depth"


using namespace DemoObject;

::p3d::P3DStatus DrainDemo::_initFromData(BIMBase::Core::BPDataCR instance)
{
	if (T_Super::_initFromData(instance) != P3DStatus::SUCCESS)
		return ERROR;

	BPValue value;
	P3DStatus status;

	status = instance.getValue(value, Property_Lenght);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setLength(value.getInteger());

	status = instance.getValue(value, Property_Width);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setWidth(value.getInteger());

	status = instance.getValue(value, Property_Thickness);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setThickness(value.getInteger());


	status = instance.getValue(value, Property_Depth);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setDepth(value.getInteger());
	return SUCCESS;
}




::p3d::P3DStatus DrainDemo::_copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const
{
	if (T_Super::_copyToData(instance, project) != P3DStatus::SUCCESS)
		return ERROR;

	P3DStatus status;
	status = instance.setValue(Property_Lenght, BPValue(this->getLength()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(Property_Width, BPValue(this->getWidth()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(Property_Thickness, BPValue(this->getThickness()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	status = instance.setValue(Property_Depth, BPValue(this->getDepth()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	return SUCCESS;
}



DrainDemo::DrainDemo()
{
	m_nWidth = 600;
	m_nThickness = 100;
	m_nDepth = 500;
	m_nLenght = 30000;
}

DrainDemo::~DrainDemo()
{

}

int DrainDemo::getWidth() const
{
	return m_nWidth;
}

void DrainDemo::setWidth(int nWidth)
{
	m_nWidth = nWidth;
}

int DrainDemo::getLength() const
{
	return m_nLenght;
}

void DrainDemo::setLength(int nLength)
{
	m_nLenght = nLength;
}

int DrainDemo::getThickness() const
{
	return m_nThickness;
}

void DrainDemo::setThickness(int nThickness)
{
	m_nThickness = nThickness;
}



int DrainDemo::getDepth() const
{
	return m_nDepth;
}

void DrainDemo::setDepth(int nDepth)
{
	m_nDepth = nDepth;
}

BIMBase::Core::BPGraphicsPtr DemoObject::DrainDemo::_createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool bIsDynamics)
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

	int nWidth = 0.5 * getWidth();
	int nLength = getLength();
	int nThickness = getThickness();
	int nDepth = getDepth();

	GeVec3d veccc = GeVec3d::create(0, nLength, 0);
	//GeVec3d veccc = GeVec3d::create(0,  0, nLength);

	Int32 nStyle = 0;
	BIMBase::BPColorDef colorGray(120, 120, 120);

	UInt32 nDrainColor = BPColorUtil::getEntityColor(colorGray, *pProject, true);


	BPSymbology sysDrain;
	sysDrain.color = nDrainColor;
	sysDrain.weight = 3;
	sysDrain.style = nStyle;


	GeCurveArrayPtr ptrOutLinesLDrain = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
	pvector<GePoint3d> ptsLDrain;
	IGeCurveBasePtr ptrLineLDrain;
	
	double dDown = -(nDepth + nThickness);
	ptsLDrain.push_back(GePoint3d::create(nWidth + nThickness, 0, dDown));
	ptsLDrain.push_back(GePoint3d::create(nWidth + nThickness, 0, nDepth + nThickness + dDown));
	ptsLDrain.push_back(GePoint3d::create(nWidth, 0, nDepth + nThickness + dDown));
	ptsLDrain.push_back(GePoint3d::create(nWidth, 0, nThickness + dDown));
	ptsLDrain.push_back(GePoint3d::create(-nWidth, 0, nThickness + dDown));
	ptsLDrain.push_back(GePoint3d::create(-nWidth, 0, nDepth + nThickness + dDown));
	ptsLDrain.push_back(GePoint3d::create(-nWidth - nThickness, 0, nDepth + nThickness + dDown));
	ptsLDrain.push_back(GePoint3d::create(-nWidth - nThickness, 0, dDown));
	ptsLDrain.push_back(GePoint3d::create(nWidth + nThickness, 0, dDown));


	ptrLineLDrain = IGeCurveBase::createLineString(ptsLDrain);
	ptrOutLinesLDrain->push_back(ptrLineLDrain);
	GeExtrusionInfo extrDataLDrain(ptrOutLinesLDrain, veccc, true);
	IGeSolidBasePtr ptrExtrusionLDrain = IGeSolidBase::createGeExtrusion(extrDataLDrain);
	ptrGraphic->addGeSolidBase(*ptrExtrusionLDrain, sysDrain);



	return ptrGraphic;
}

BIMBase::Core::BPGraphicsPtr  DemoObject::DrainDemo::_createPhysicalGraphicsForDrawing(BIMBase::Core::BPProject& project, BIMBase::PModelIdCR modelId) {
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return nullptr;
	BPModelPtr ptrModel = project.getModelById(modelId);
	if (ptrModel.isNull())
		return nullptr;

	BPGraphicsPtr ptrGraphic = ptrModel->createPhysicalGraphics();
	if (ptrGraphic.isNull())
		return nullptr;


	int nWidth = 0.5 * getWidth();
	int nLength = getLength();
	int nThickness = getThickness();
	int nDepth = getDepth();


	Int32 nStyle = 0;
	BIMBase::BPColorDef colorGray(120, 120, 120);

	UInt32 nDrainColor = BPColorUtil::getEntityColor(colorGray, *pProject, true);


	BPSymbology sysDrain;
	sysDrain.color = nDrainColor;
	sysDrain.weight = 3;
	sysDrain.style = nStyle;

	GeCurveArrayPtr ptrOutLinesLDrain = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
	pvector<GePoint3d> ptsLDrain, ptsMidLine, ptsTopLine;
	IGeCurveBasePtr ptrLineLDrain, ptrMidLineL, ptrTopLine;
	ptsLDrain.push_back(GePoint3d::create(nWidth + nThickness, 0, 0));
	ptsLDrain.push_back(GePoint3d::create(nWidth + nThickness, nDepth + nThickness, 0));
	ptsLDrain.push_back(GePoint3d::create(nWidth, nDepth + nThickness, 0));
	ptsLDrain.push_back(GePoint3d::create(nWidth, nThickness, 0));
	ptsLDrain.push_back(GePoint3d::create(-nWidth, nThickness, 0));
	ptsLDrain.push_back(GePoint3d::create(-nWidth, nDepth + nThickness, 0));
	ptsLDrain.push_back(GePoint3d::create(-nWidth - nThickness, nDepth + nThickness, 0));
	ptsLDrain.push_back(GePoint3d::create(-nWidth - nThickness, 0, 0));
	ptsLDrain.push_back(GePoint3d::create(nWidth + nThickness, 0, 0));

	ptsMidLine.push_back(GePoint3d::create(0, 0.3 * (nDepth + nThickness), 0));
	ptsMidLine.push_back(GePoint3d::create(0, 1.5 * (nDepth + nThickness), 0));

	ptsTopLine.push_back(GePoint3d::create(0.1 * (nWidth + nThickness), 1.3 * (nDepth + nThickness), 0));
	ptsTopLine.push_back(GePoint3d::create(0, 1.5 * (nDepth + nThickness), 0));
	ptsTopLine.push_back(GePoint3d::create(-0.1 * (nWidth + nThickness), 1.3 * (nDepth + nThickness), 0));

	ptrLineLDrain = IGeCurveBase::createLineString(ptsLDrain);
	ptrMidLineL = IGeCurveBase::createLineString(ptsMidLine);
	ptrTopLine = IGeCurveBase::createLineString(ptsTopLine);
	ptrOutLinesLDrain->push_back(ptrLineLDrain);
	ptrOutLinesLDrain->push_back(ptrMidLineL);
	ptrOutLinesLDrain->push_back(ptrTopLine);
	ptrGraphic->addGeCurveArray(*ptrOutLinesLDrain);

	return ptrGraphic;
}





