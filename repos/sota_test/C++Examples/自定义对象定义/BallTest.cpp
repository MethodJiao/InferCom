#include "stdafx.h"
#include "BallTest.h"

#define Property_Origin                          "Origin"


using namespace TestObject;

BallTest::BallTest()
{
}

BallTest::~BallTest()
{

}

::p3d::P3DStatus      BallTest::_copyToData(::BIMBase::Core::BPDataR data, ::BIMBase::Core::BPProjectR project) const
{
	if (T_Super::_copyToData(data, project) != P3DStatus::SUCCESS)
		return ERROR;

	P3DStatus status;
	status = data.setValue(Property_Origin, BPValue(m_pOrigin));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	return SUCCESS;
}

::p3d::P3DStatus  BallTest::_initFromData(::BIMBase::Core::BPDataCR data)
{
	if (T_Super::_initFromData(data) != P3DStatus::SUCCESS)
		return ERROR;

	BPValue value;
	P3DStatus status;

	status = data.getValue(value, Property_Origin);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setOrigin(value.getPoint3D());


	return SUCCESS;
}

BIMBase::Core::BPGraphicsPtr    BallTest::_createPhysicalGraphics(::BIMBase::Core::BPProjectR project, ::BIMBase::PModelIdCR modelId, bool isDynamics)
{
	BPModelPtr ptrModel = project.getModelById(modelId);
	if (ptrModel.isNull())
		return nullptr;

	BPGraphicsPtr ptrPhysicalGeometry = ptrModel->createPhysicalGraphics();
	if (ptrPhysicalGeometry.isNull())
		return nullptr;

	GeSphereInfo sphere(m_pOrigin, 500);
	IGeSolidBasePtr ptrExtrusion = IGeSolidBase::createGeSphere(sphere);
	ptrPhysicalGeometry->addGeSolidBase(*ptrExtrusion);

	return ptrPhysicalGeometry;
}
