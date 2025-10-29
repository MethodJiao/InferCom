#include "stdafx.h"
#include "SolidDemo.h"

#define Property_Points                         "Points"
#define Property_Type                           "SolidType"

using namespace DemoObject;

SolidDemo::SolidDemo()
{
	m_solidType = GeSolidBaseType::GeSolidBaseType_TorusPipe;
}

SolidDemo::~SolidDemo()
{

}

::p3d::P3DStatus SolidDemo::_copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const
{
	if (T_Super::_copyToData(instance, project) != P3DStatus::SUCCESS)
		return ERROR;

	P3DStatus status;
	status = instance.setValue(Property_Type, (BPValue)(int)(this->getSolidType()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;
 
	instance.clearArray(Property_Points);
	status = instance.addArrayElements(Property_Points, m_points.size());
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	for (int i=0; i<m_points.size(); i++)
	{
		status = instance.setValue(Property_Points, (BPValue)m_points.at(i), i);
		if (P3DStatus::SUCCESS != status)
			return ERROR;
	}

	return SUCCESS;
}

::p3d::P3DStatus SolidDemo::_initFromData(BIMBase::Core::BPDataCR instance)
{
	if (T_Super::_initFromData(instance) != P3DStatus::SUCCESS)
		return ERROR;
	m_points.clear();
	BPValue value;
	P3DStatus status;

	status = instance.getValue(value, Property_Type);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setSolidType((GeSolidBaseType)value.getInteger());

	status = instance.getValue(value, Property_Points);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	int nArrayCount = value.getArrayInfo().GetCount();

	for (int i = 0; i < nArrayCount; i++)
	{
		status = instance.getValue(value, Property_Points, i);
		if (P3DStatus::SUCCESS != status)
			return ERROR;
		m_points.push_back(value.getPoint3D());
	}

	return SUCCESS;
}

BIMBase::Core::BPGraphicsPtr DemoObject::SolidDemo::_createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool bIsDynamics)
{
	BPModelPtr ptrModel = project.getModelById(modelId);
	if (ptrModel.isNull())
		return nullptr;

	BPGraphicsPtr ptrGraphic = ptrModel->createPhysicalGraphics();
	switch (m_solidType)
	{
	case p3d::GeSolidBaseType::GeSolidBaseType_None:
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_TorusPipe:
		createTorusPipe(ptrGraphic);
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_Cone:
		createCone(ptrGraphic);
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_Box:
		createBox(ptrGraphic);
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_Sphere:
		createSphere(ptrGraphic);
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_Extrusion:
		createExtrusion(ptrGraphic);
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_RotationalSweep:
		createRotationSweep(ptrGraphic);
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_RuledSweep:
		createRuledSweep(ptrGraphic);
		break;
	default:
		break;
	}

	return ptrGraphic;
}

BIMBase::BPSnapStatus DemoObject::SolidDemo::_onSnap(BIMBase::Core::BPSnapContext& snapContext)
{
	if (m_solidType != p3d::GeSolidBaseType::GeSolidBaseType_Sphere)
		return BPSnapStatus::eSuccess;

	BPSnapMode snapMode = snapContext.getSnapMode();

	p3d::platform::BPSnapPathImpP pSnapPath = snapContext.getSnapPath();
	if (NULL == pSnapPath)
		return BPSnapStatus::eDisabled;

	//forceHot为false时，只有当用户将鼠标靠近捕捉点的位置，才会捕捉到该点；
	//forceHor为true时，不论鼠标与捕捉点的距离多远，都能捕捉到该点。
	bool bForceHot = false;
	GePoint3d hitPoint;

	switch (snapMode)
	{
	case BPSnapMode::Center:
	{
		//获取长度方向的中点
		GePoint3d ptCenter = m_points[0];
		//局部坐标系转到世界坐标系下
		GeTransform trans = getPlacement().toTransform();
		trans.multiply(ptCenter);
		hitPoint = ptCenter;
	}
	break;
	}

	//捕捉点设置
	snapContext.setSnapInfo(0, snapMode, hitPoint, bForceHot, true, 0, NULL);

	return BPSnapStatus::eSuccess;
}

void DemoObject::SolidDemo::setPoints(std::vector<GePoint3d>& points)
{
	if (points.size() == 0)
		return;

	m_points.clear();
	GeTransform trans = GeTransform::create(points[0]);
	BPPlacement placement;
	placement.fromTransform(trans);
	setPlacement(placement);

	for each (auto point in points)
	{
		m_points.push_back(point - points[0]);
	}
}

bool DemoObject::SolidDemo::createTorusPipe(BPGraphicsPtr ptrGraphic)
{
	switch (m_points.size())
	{
	case 1:
		return false;
		break;
	case 2:
	{
		IGeCurveBasePtr ptrCurve = IGeCurveBase::createSegment(GeSegment3d::create(m_points[0], m_points[1]));
		ptrGraphic->addGeCurve(*ptrCurve);
	}
		break;
	case 3:
	{
		IGeCurveBasePtr ptrCurve = IGeCurveBase::createSegment(GeSegment3d::create(m_points[0], m_points[1]));
		ptrGraphic->addGeCurve(*ptrCurve);
		GeVec3d normal = GeVec3d::create(0, 0, 1) ^ (m_points[1] - m_points[0]);
		IGeCurveBasePtr ptrCurveElli = IGeCurveBase::createEllipse(GeEllipse3d::createByCircle(m_points[1], normal, m_points[1].distance(m_points[2])));
		ptrGraphic->addGeCurve(*ptrCurveElli);
	}
		break;
	case 4:
	{
		GeVec3d vctS = m_points[3] - m_points[0];
		GeVec3d vctE = m_points[1] - m_points[0];
		double angel = vctE.angleTo2D(vctS);
		double minR = m_points[1].distance(m_points[2]);
		GeVec3d normal = GeVec3d::create(0, 0, 1) ^ (m_points[1] - m_points[0]);
		vctE.normalize();
		normal.normalize();
		GeTorusPipeInfo pipeInfo(m_points[0],
			vctE,
			normal,
			m_points[0].distance(m_points[1]),
			minR,
			angel,
			true);
		IGeSolidBasePtr ptrSolid = IGeSolidBase::createGeTorusPipe(pipeInfo);
		ptrGraphic->addGeSolidBase(*ptrSolid);
	}
		break;
	}
	return true;
}

bool DemoObject::SolidDemo::createCone(BPGraphicsPtr ptrGraphic)
{
	switch (m_points.size())
	{
	case 1:
		return false;
		break;
	case 2:
	{
		IGeCurveBasePtr ptrCurve = IGeCurveBase::createEllipse(GeEllipse3d::createByCircle(m_points[0], GeVec3d::create(0, 0, 1), m_points[1].distance(m_points[0])));
		ptrGraphic->addGeCurve(*ptrCurve);
	}
	break;
	case 3:
	{
		GeConeInfo coneInfo(m_points[0], m_points[2], m_points[1].distance(m_points[0]), 0, TRUE);
		IGeSolidBasePtr ptrSolid = IGeSolidBase::createGeCone(coneInfo);
		ptrGraphic->addGeSolidBase(*ptrSolid);
	}
	break;
	}
	return true;
}

bool DemoObject::SolidDemo::createBox(BPGraphicsPtr ptrGraphic)
{
	GePoint3d pbtmOrigin = GePoint3d::create(0, 0, 0);
	GePoint3d ptopOrigin = GePoint3d::create(2000, 0, 4000);
	GeVec3d   vecX = GeVec3d::create(1, 0, 0);
	GeVec3d   vecY = GeVec3d::create(0, 1, 0);
	GeBoxInfo boxInfo(pbtmOrigin, ptopOrigin, vecX, vecY, 5000,7000,4000,2000, true);
	IGeSolidBasePtr ptrSolid = IGeSolidBase::createGeBox(boxInfo);
	ptrGraphic->addGeSolidBase(*ptrSolid);
	return true;
}

bool DemoObject::SolidDemo::createSphere(BPGraphicsPtr ptrGraphic)
{
	if (m_points.size() < 2)
		return false;
	GeSphereInfo sphereInfo(GePoint3d::create(0, 0, 0), m_points[1].distance(m_points[0]));
	IGeSolidBasePtr ptrSolid = IGeSolidBase::createGeSphere(sphereInfo);
	ptrGraphic->addGeSolidBase(*ptrSolid);
	return true;
}

bool DemoObject::SolidDemo::createExtrusion(BPGraphicsPtr ptrGraphic)
{
	switch (m_points.size())
	{
	case 1:
		return false;
		break;
	case 2:
	{
		IGeCurveBasePtr ptrCurve = IGeCurveBase::createEllipse(GeEllipse3d::createByCircle(m_points[0], GeVec3d::create(0, 0, 1), m_points[1].distance(m_points[0])));
		ptrGraphic->addGeCurve(*ptrCurve);
	}
	break;
	case 3:
	{
		IGeCurveBasePtr ptrCurve = IGeCurveBase::createEllipse(GeEllipse3d::createByCircle(m_points[0], GeVec3d::create(0, 0, 1), m_points[1].distance(m_points[0])));
		ptrGraphic->addGeCurve(*ptrCurve);
		IGeCurveBasePtr ptrCurve2 = IGeCurveBase::createEllipse(GeEllipse3d::createByCircle(m_points[0], GeVec3d::create(0, 0, 1), m_points[2].distance(m_points[0])));
		ptrGraphic->addGeCurve(*ptrCurve2);
	}
	break;
	case 4:
	{
		IGeCurveBasePtr ptrCurve = IGeCurveBase::createEllipse(GeEllipse3d::createByCircle(m_points[0], GeVec3d::create(0, 0, 1), m_points[1].distance(m_points[0])));
		IGeCurveBasePtr ptrCurve2 = IGeCurveBase::createEllipse(GeEllipse3d::createByCircle(m_points[0], GeVec3d::create(0, 0, 1), m_points[2].distance(m_points[0])));
		GeCurveArrayPtr curveOutter = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
		GeCurveArrayPtr curveInner = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Inner);
		GeCurveArrayPtr PtrRegion = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_ParityRegion);
		curveOutter->add(ptrCurve2);
		curveInner->add(ptrCurve);
		PtrRegion->add(curveOutter);
		PtrRegion->add(curveInner);
		GeExtrusionInfo extrusionInfo(PtrRegion, GeVec3d::create(0,0,m_points[3].distance(m_points[2])), true);
		IGeSolidBasePtr ptrSolid = IGeSolidBase::createGeExtrusion(extrusionInfo);
		ptrGraphic->addGeSolidBase(*ptrSolid);
	}
	break;
	}
	return true;
}

bool DemoObject::SolidDemo::createRotationSweep(BPGraphicsPtr ptrGraphic)
{
	if (m_points.size() > 2)
	{
		GeVec3d vec2 = m_points[2] - m_points[0];
		GeVec3d vec1 = m_points[1] - m_points[0];
		double dLen1 = vec1.normalize();
		double dLen2 = 0;
		vec2.projectToVector(dLen2, vec1);
		if (dLen2 < dLen1)
			return false;
		m_points[2] = m_points[0] + vec1 * dLen2;
	}
	switch (m_points.size())
	{
	case 1:
		return false;
		break;
	case 2:
	{
		IGeCurveBasePtr ptrCurve = IGeCurveBase::createSegment(GeSegment3d::create(m_points[0], m_points[1]));
		ptrGraphic->addGeCurve(*ptrCurve);
		GePoint3d pointTop = GePoint3d::create(m_points[0].x, m_points[0].y, m_points[1].distance(m_points[0]));
		IGeCurveBasePtr ptrCurve2 = IGeCurveBase::createEllipse(GeEllipse3d::createByEllipseCenterStartEnd(m_points[0], m_points[1], pointTop));
		ptrGraphic->addGeCurve(*ptrCurve2);
	}
	break;
	case 3:
	{
		GePoint3d pointTopTop = GePoint3d::create(m_points[0].x, m_points[0].y, m_points[2].distance(m_points[0]));
		GePoint3d pointTop = GePoint3d::create(m_points[0].x, m_points[0].y, m_points[1].distance(m_points[0]));
		IGeCurveBasePtr ptrCurveOut = IGeCurveBase::createEllipse(GeEllipse3d::createByEllipseCenterStartEnd(m_points[0], m_points[2], pointTopTop));
		IGeCurveBasePtr ptrCurve1 = IGeCurveBase::createSegment(GeSegment3d::create(pointTopTop, pointTop));
		IGeCurveBasePtr ptrCurvetemp = IGeCurveBase::createEllipse(GeEllipse3d::createByEllipseCenterStartEnd(m_points[0], m_points[1], pointTop));
		GePoint3d point;
		ptrCurvetemp->proportToPoint(0.5, point);
		IGeCurveBasePtr ptrCurveIn = IGeCurveBase::createEllipse(GeEllipse3d::createByPointsOnEllipse(pointTop, point,m_points[1]));
		IGeCurveBasePtr ptrCurve2 = IGeCurveBase::createSegment(GeSegment3d::create(m_points[1], m_points[2]));
		GeCurveArrayPtr ptrCurveArray = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
		ptrCurveArray->add(ptrCurveOut);
		ptrCurveArray->add(ptrCurve1);
		ptrCurveArray->add(ptrCurveIn);
		ptrCurveArray->add(ptrCurve2);
		ptrGraphic->addGeCurveArray(*ptrCurveArray);
	}
	break;
	case 4:
	{
		GePoint3d pointTopTop = GePoint3d::create(m_points[0].x, m_points[0].y, m_points[2].distance(m_points[0]));
		GePoint3d pointTop = GePoint3d::create(m_points[0].x, m_points[0].y, m_points[1].distance(m_points[0]));
		IGeCurveBasePtr ptrCurveOut = IGeCurveBase::createEllipse(GeEllipse3d::createByEllipseCenterStartEnd(m_points[0], m_points[2], pointTopTop));
		IGeCurveBasePtr ptrCurve1 = IGeCurveBase::createSegment(GeSegment3d::create(pointTopTop, pointTop));
		IGeCurveBasePtr ptrCurvetemp = IGeCurveBase::createEllipse(GeEllipse3d::createByEllipseCenterStartEnd(m_points[0], m_points[1], pointTop));
		GePoint3d point;
		ptrCurvetemp->proportToPoint(0.5, point);
		IGeCurveBasePtr ptrCurveIn = IGeCurveBase::createEllipse(GeEllipse3d::createByPointsOnEllipse(pointTop, point, m_points[1]));
		IGeCurveBasePtr ptrCurve2 = IGeCurveBase::createSegment(GeSegment3d::create(m_points[1], m_points[2]));
		GeCurveArrayPtr ptrCurveArray = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
		ptrCurveArray->add(ptrCurveOut);
		ptrCurveArray->add(ptrCurve1);
		ptrCurveArray->add(ptrCurveIn);
		ptrCurveArray->add(ptrCurve2);

		GeVec3d vctS = m_points[3] - m_points[0];
		GeVec3d vctE = m_points[2] - m_points[0];
		double angel = vctE.angleTo2D(vctS);
		double minR = m_points[1].distance(m_points[2]);
		GeRotationalSweepInfo sweepInfo(ptrCurveArray, m_points[0], GeVec3d::create(0,0,1), angel, true);
		IGeSolidBasePtr ptrSolid = IGeSolidBase::createGeRotationalSweep(sweepInfo);
		ptrGraphic->addGeSolidBase(*ptrSolid);
	}
	break;
	}
	return true;
}

bool DemoObject::SolidDemo::createRuledSweep(BPGraphicsPtr ptrGraphic)
{
	GePoint3d point0 = GePoint3d::create(2000,0,0);
	pvector<GePoint3d> pList;
	vector<GePoint3d> pList2;
	GeTransform trans = GeTransform::createByAxisAndRotationAngle(GeRay3d::createByOriginAndVector(GePoint3d::create(0, 0, 0), GeVec3d::create(0, 0, 1)), PI / 3);
	for (int i = 0; i < 6; i++)
	{
		pList.push_back(point0);
		pList2.push_back(point0);
		point0 = GePoint3d::createByTransform(trans, point0);
	}
	GeCurveArrayPtr ptrCurvearray = GeCurveArray::createLinestringArray(pList, GeCurveArray::BOUNDARY_TYPE_Outer, true);
	
	pvector<GeCurveArrayPtr> sectionCurves;
	sectionCurves.push_back(ptrCurvearray);
	for (int i = 1; i < m_points.size(); i++)
	{
		trans = GeTransform::createByAxisAndRotationAngle(GeRay3d::createByOriginAndVector(GePoint3d::create(0, 0, 0), GeVec3d::create(0, 0, 1)), PI / 3*i);
		GeTransform transform = GeTransform::create(GeVec3d::create(0, 0, m_points[i].distance(m_points[0])));
		GeCurveArrayPtr ptrCurvearray2 = ptrCurvearray->clone();
		ptrCurvearray2->setByTransform(GeTransform::createByProduct(transform, trans));
		sectionCurves.push_back(ptrCurvearray2);
	}


	GeRuledSweepInfo RuledInfo(sectionCurves, true);
	IGeSolidBasePtr ptrSolid = IGeSolidBase::createGeRuledSweep(RuledInfo);
	ptrGraphic->addGeSolidBase(*ptrSolid);		
	return true;
}

