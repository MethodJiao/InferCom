#include "stdafx.h"
#include "TubeDemo.h"


using namespace DemoObject;
using namespace p3d;

#define PROPERTY_TUBEPOSITIONS  "tubePositions"
#define PROPERTY_TUBEDIAMETER  "tubeDiameter"
#define PROPERTY_TUBETHICKNESS "tubeThickness"


Tube::Tube(GePoint3d ptCenter, double dTubeDiameter, double dTubeThickness)
{
	m_ptCenter = ptCenter;
	m_dTubeDiameter = dTubeDiameter;
	m_dTubeThickness = dTubeThickness;
}

Tube::Tube()
{
	m_ptCenter = GePoint3d::createByZero();
	m_dTubeDiameter = 280;
	m_dTubeThickness = 10;
}

bool Tube::setCenter(GePoint3d ptCenter)
{
	m_ptCenter = ptCenter;
	return true;
}

GePoint3d Tube::getCenter() const
{
	return m_ptCenter;
}

bool Tube::setDiameter(double dTubeDiameter)
{
	if (dTubeDiameter < 0)
		return false;
	m_dTubeDiameter = dTubeDiameter;
	return true;
}

double Tube::getDiameter() const
{
	return m_dTubeDiameter;
}

bool Tube::setThickness(double dTubeThickness)
{
	if (dTubeThickness < 0)
		return false;
	m_dTubeThickness = dTubeThickness;
	return true;
}

double Tube::getThickness() const
{
	return m_dTubeThickness;
}

::p3d::P3DStatus Tube::_initFromData(BIMBase::Core::BPDataCR instance)
{
	if (T_Super::_initFromData(instance) != P3DStatus::SUCCESS)
		return ERROR;

	BPValue value;
	P3DStatus status;

	status = instance.getValue(value, PROPERTY_TUBEPOSITIONS);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setCenter(value.getPoint3D());

	status = instance.getValue(value, PROPERTY_TUBEDIAMETER);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setDiameter(value.getDouble());

	status = instance.getValue(value, PROPERTY_TUBETHICKNESS);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setThickness(value.getDouble());

	return SUCCESS;
}

::p3d::P3DStatus Tube::_copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const
{
	if (T_Super::_copyToData(instance, project) != P3DStatus::SUCCESS)
		return ERROR;

	P3DStatus status;
	status = instance.setValue(PROPERTY_TUBEPOSITIONS, BPValue(getCenter()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(PROPERTY_TUBEDIAMETER, BPValue(getDiameter()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(PROPERTY_TUBETHICKNESS, BPValue(getThickness()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	return SUCCESS;
}

TubePtr Tube::deepClone()
{
	if (!this)
		return nullptr;
	TubePtr ptrNewTube = new Tube();
	ptrNewTube->setCenter(m_ptCenter);
	ptrNewTube->setDiameter(m_dTubeDiameter);
	ptrNewTube->setThickness(m_dTubeThickness);
	return ptrNewTube;
}
