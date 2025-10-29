#include "stdafx.h"
#include "CreatePolyfaceDemo.h"


CreatePolyfaceDemo::CreatePolyfaceDemo()
{
}


CreatePolyfaceDemo::~CreatePolyfaceDemo()
{
}
//һ�����Ƶ���
void CreatePolyfaceDemo::funPolyface()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;

	BPModelBaseP ptrModel = pProject->getActiveModel();
	if (ptrModel == nullptr)
		return;
	pvector<GePoint3d> points1;
	pvector<GePoint3d> points2;
	GePoint3d pts = GePoint3d::create(0,0,0);

	GePoint3d pts1 = GePoint3d::create(100, 50, 100);
	GePoint3d pts2 = GePoint3d::create(150, 50, 200);
	GePoint3d pts3 = GePoint3d::create(200, 100, 250);
	GePoint3d pts4 = GePoint3d::create(200, 200, 200);
	GePoint3d pts5 = GePoint3d::create(150, 200, 200);
	GePoint3d pts6 = GePoint3d::create(100, 200, 100);
	GePoint3d pts7 = GePoint3d::create(0, 100, 0);

	/*GePoint3d pts4 = GePoint3d::create(400, 100,250);
	GePoint3d pts5 = GePoint3d::create(350, 200, 250);*/
	GePoint3d pts31 = GePoint3d::create(200, 100, 250);
	GePoint3d pts8 = GePoint3d::create(250, 120, 300);
	GePoint3d pts9 = GePoint3d::create(350, 150, 350);
	GePoint3d pts10 = GePoint3d::create(400, 100, 350);
	GePoint3d pts11 = GePoint3d::create(400, 200, 300);
	GePoint3d pts12 = GePoint3d::create(350, 200, 250);
	GePoint3d pts13 = GePoint3d::create(200, 200, 200);



	points1.push_back(pts);
	points1.push_back(pts1);
	points1.push_back(pts2);
	points1.push_back(pts3);
	points1.push_back(pts4);
	points1.push_back(pts5);
	points1.push_back(pts6);
	points1.push_back(pts7);

	points2.push_back(pts31);
	points2.push_back(pts8);
	points2.push_back(pts9);
	points2.push_back(pts10);
	points2.push_back(pts11);
	points2.push_back(pts12);
	points2.push_back(pts13);
	

	PolyfaceHandlePtr ptrMesh = PolyfaceHandle::createVariableSizeIndexed();
	PolyfaceHandlePtr ptrMesh1 = PolyfaceHandle::createVariableSizeIndexed();
	ptrMesh->addPolygon(points1);
	ptrMesh1->addPolygon(points2);
	BPGraphicsPtr ptrGrapic = ptrModel->createPhysicalGraphics();
	if (ptrGrapic.isNull())
		return;
	ptrGrapic->addPolyface(*ptrMesh);
	ptrGrapic->addPolyface(*ptrMesh1);
	ptrGrapic->save();

}

void cacuPolygon(pvector<pvector<GePoint3d>>& Triangles, pvector<GePoint3d>& curvestart, pvector<GePoint3d>& curveEnd)
{
	Triangles.clear();
	int nn = curvestart.size();
	for (int i = 0; i < nn; i++)
	{
		int nNext = (i + 1) % nn;
		pvector<GePoint3d> vctPointsTriangle1, vctPointsTriangle2;
		vctPointsTriangle1.push_back(curvestart[i]);
		vctPointsTriangle1.push_back(curvestart[nNext]);
		vctPointsTriangle1.push_back(curveEnd[nNext]);

		vctPointsTriangle2.push_back(curveEnd[nNext]);
		vctPointsTriangle2.push_back(curveEnd[i]);
		vctPointsTriangle2.push_back(curvestart[i]);

		Triangles.push_back(vctPointsTriangle1);
		Triangles.push_back(vctPointsTriangle2);
	}
}

BPGraphicsPtr CreatePolyfaceDemo::createPolyfaceSolid()
{
	//�ı���������������ת10��
	PolyfaceHandlePtr ptrPolyface = PolyfaceHandle::createNewOne();//������ת����

	double dRotAngle = double(10) / 180.0 * PI;
	double dPolyAngle = PI - PI * ((double)4 - 2.0) / (double)4;

	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return nullptr;
	BPModelBaseP pModel = pProject->getActiveModel();
	BPGraphicsPtr ptrGraphic = pModel->createPhysicalGraphics();

	//�������ε�
	pvector<GePoint3d> vctPoints, vctPointsDown, vctPointsUp2;
	GePoint3d pEdgeStart = GePoint3d::createByZero();
	GePoint3d pEdgeEnd = GePoint3d::create(1000, 0, 0);
	vctPoints.push_back(pEdgeStart);
	vctPoints.push_back(pEdgeEnd);
	vctPointsDown.reserve(4);
	vctPointsDown.push_back(pEdgeStart);
	GeVec3d vecEdge = pEdgeEnd - pEdgeStart;
	for (int i = 0; i < 4 - 2; i++)
	{
		vecEdge.rotate2D(dPolyAngle);
		vctPoints.push_back(vctPoints.back() + vecEdge);
	}
	for (int i = 1; i < vctPoints.size(); i++)
	{
		vctPointsDown.push_back(vctPoints[vctPoints.size() - i]);
	}
	GeCurveArrayPtr ptrCurveFaceDown = GeCurveArray::createLinestringArray(vctPointsDown, GeCurveArray::BOUNDARY_TYPE_Outer);
	ptrPolyface->addPolygon(vctPointsDown);

	//�������ζ�
	GeTransform transform1 = GeTransform::create(GePoint3d::create(0, 0, 1000));
	GePoint3d centroid = GePoint3d::createByZero();
	GeVec3d normal = GeVec3d::create(0, 0, 0);
	double dArea = 0;
	ptrCurveFaceDown->getCentroidAndArea2D(centroid, dArea);
	GeTransform transRotate = GeTransform::createByProduct(GeTransform::create(GeRotMatrix::createByAxisAndRotationAngle(2, dRotAngle)), GeTransform::create(centroid * (-1)));
	GeTransform transOrigin = GeTransform::createByProduct(GeTransform::create(centroid), transRotate);
	GeTransform trans = GeTransform::createByProduct(transform1, transOrigin);
	for (int i = 0; i < 4; i++)
	{
		GePoint3d pointTemp2 = GePoint3d::createByTransform(trans, vctPoints[i]);
		vctPointsUp2.push_back(pointTemp2);
	}
	GeCurveArrayPtr ptrCurveFaceUp = GeCurveArray::createLinestringArray(vctPointsUp2, GeCurveArray::BOUNDARY_TYPE_Outer);
	ptrPolyface->addPolygon(vctPointsUp2);

	//�������β���
	pvector<pvector<GePoint3d>>  vctvctPoints;
	cacuPolygon(vctvctPoints, vctPoints, vctPointsUp2);
	for (int i = 0; i < vctvctPoints.size(); i++)
	{
		ptrPolyface->addPolygon(vctvctPoints[i]);
	}

	ptrGraphic->addPolyface(*ptrPolyface);
	return ptrGraphic;
}



void CreatePolyfaceDemo::funCreatePolyfaceSolid()
{
	BPGraphicsPtr ptrGraphic = CreatePolyfaceDemo::createPolyfaceSolid();
	if (ptrGraphic.isNull())
		return;
	ptrGraphic->save();
}

void CreatePolyfaceDemo::doBoolean()
{
	//Բ����
	GeConeInfo coneInfo(GePoint3d::create(2000, 600, 500), GePoint3d::create(-300, 600, 500),100,100,true);

	IGeSolidBasePtr ptrSolid = IGeSolidBase::createGeCone(coneInfo);
	if (ptrSolid.isNull())
		return;

	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	BPModelBaseP pModel = pProject->getActiveModel();
	BPGraphicsPtr ptrGraphicCone = pModel->createPhysicalGraphics();

	if ( ptrGraphicCone.isNull())
		return;

	ptrGraphicCone->addGeSolidBase(*ptrSolid);

	//��ȡpolyface
	BPGraphicsPtr ptrGraphicPoly = createPolyfaceSolid();
	if (ptrGraphicPoly.isNull())
		return;

	//����,Բ���ü�polyface
	BPGraphicsPtr ptrGraphicResult = pModel->createPhysicalGraphics();
	int tolerance = 0;
	BPSolidBooleanUtil::getAngleTolerance(tolerance);
	BPSolidBooleanUtil::setAngleTolerance(36);//����ǰ����������Ϊ36����ֹ����������Ƭ����
	BPSolidBooleanUtil::doBoolean(ptrGraphicResult, ptrGraphicPoly, ptrGraphicCone, BPBooleanOp::Substract);
	BPSolidBooleanUtil::setAngleTolerance(tolerance);//����������ؾ���
	ptrGraphicResult->save();

	//����һ���µ�Բ����polyface
	BPGraphicsPtr ptrGraphicConeNew;
	ptrGraphicConeNew = pModel->createPhysicalGraphics();
	BPGraphicsUtils::copyPhysicalGraphics(*ptrGraphicConeNew, *ptrGraphicCone);
	if (ptrGraphicConeNew.isNull())
		return;
	BPGraphicsUtils::transformPhysicalGraphics(*ptrGraphicConeNew, GeTransform::create(GePoint3d::create(5000, 0, 0)));
	BPGraphicsPtr ptrGraphicPolyNew = createPolyfaceSolid();
	if (ptrGraphicPolyNew.isNull())
		return;
	BPGraphicsUtils::transformPhysicalGraphics(*ptrGraphicPolyNew, GeTransform::create(GePoint3d::create(5000, 0, 0)));

	//������polyface�ü�Բ��������ֻҪԲ����������Ұ��--------------------------------------
	BPGraphicsPtr ptrGraphicResultNew = pModel->createPhysicalGraphics();
	BPGraphicsPtr ptrGraphicModify = pModel->createPhysicalGraphics();
	BPSolidBooleanUtil::setAngleTolerance(36);//����ǰ����������Ϊ36����ֹ����������Ƭ����
	BPSolidBooleanUtil::doBoolean(ptrGraphicResultNew, ptrGraphicConeNew, ptrGraphicPolyNew, BPBooleanOp::Substract);
	BPSolidBooleanUtil::setAngleTolerance(tolerance);//����������ؾ���
	if (ptrGraphicResultNew.isNull())
		return;
	if (ptrGraphicModify.isNull())
		return;
	for (BPGraphics::EntryPtr& loadedEntry : *ptrGraphicResultNew)
	{
		switch (loadedEntry->getType())
		{
		case BPGraphics::Entry::Type::Polyface:
		{
			PolyfaceHandleP pPolyface = loadedEntry->getAsPolyfaceHandleP();
			if (pPolyface == nullptr)
				continue;

			p3d::pvector<p3d::PolyfaceHandlePtr> vctPolyface;
			BPMeshMeans::separateMeshByTopology(vctPolyface, pPolyface);

			for (auto value : vctPolyface)
			{
				if (value.isNull())
					continue;

				TemplateVectorGePoint3dR vctPoints = value->getPointR();
				TemplateVectorIntR vctPtIndex = value->getPointIndexR();
				bool bNeed = false;

				//���ݵ�������ж���Ҫ�Ĳ���
				for (auto index : vctPtIndex)
				{
					if (index >= vctPoints.size() || index < 0)
						continue;

					if (vctPoints[index].x > 6000)
					{
						bNeed = true;
						break;
					}
				}

				if (bNeed)
				{
					ptrGraphicModify->addPolyface(*value);
				}
			}
		}
		}
	}

	ptrGraphicModify->save();
}

bool CreatePolyfaceDemo::solidBaseToPolyface(pvector<PolyfaceHandlePtr> meshData, IGeSolidBasePtr solidPrimitive)
{
	PPCGraphics temp;
	temp.add(solidPrimitive);
	return SolidCore::BPMeshMeans::convertToMesh(meshData, temp.get(), 36);
}

void CreatePolyfaceDemo::combinePolyface()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (!pProject)
		return;
	BPModelBaseP pModel = pProject->getActiveModel();
	PolyfaceHandlePtr ptrPolyface = PolyfaceHandle::createNewOne();
	BPGraphicsPtr ptrGraphic = pModel->createPhysicalGraphics();
	BPGraphicsPtr ptrGraphic2 = pModel->createPhysicalGraphics();

	pvector<GePoint3d> vctPoints{};
	vctPoints.push_back({ 0,0,1000 });
	vctPoints.push_back({ 1000,0,1000 });
	vctPoints.push_back({ 1000,1000,1000 });

	ptrPolyface->addPolygon(vctPoints);
	ptrGraphic->addPolyface(*ptrPolyface);
	//if (ptrGraphic.isNull())
	//	return;
	//ptrGraphic->save();

	auto ptPolyface = ptrPolyface->getPointCP();
	auto indexPolyface = ptrPolyface->getPointIndexCP();
	auto ptPolyfaceCount = ptrPolyface->getPointCount();
	auto indexPolyfaceCount = ptrPolyface->getPointIndexCount();
	vector<GePoint3d> vctPtPolyface;
	vector<int> vctIndexPolyface;

	TemplateVectorGePoint3dR ptR = ptrPolyface->getPointR();
	TemplateVectorIntR pointIndexesR = ptrPolyface->getPointIndexR();


	for (size_t i = 0; i < ptPolyfaceCount; i++)
	{
		vctPtPolyface.push_back(ptPolyface[i]);
	}
	for (size_t i = 0; i < indexPolyfaceCount; i++)
	{
		vctIndexPolyface.push_back(indexPolyface[i]);
	}
	
	ptR.push_back({ 0,1000,1000 });
	pointIndexesR.clear();
	pointIndexesR.push_back(1);
	pointIndexesR.push_back(2);
	pointIndexesR.push_back(3);
	pointIndexesR.push_back(1);
	pointIndexesR.push_back(4);
	pointIndexesR.push_back(3);
	pointIndexesR.push_back(0);
	//ԭ��˳��Ϊ1230���˴�������143
	ptrPolyface->combineCoordinate();
	ptrGraphic2->addPolyface(*ptrPolyface);
	if (ptrGraphic2.isNull())
		return;
	ptrGraphic2->save();
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("polyfaceDemo", CreatePolyfaceDemo::funPolyface);
BPToolsManager::registerFun("polyfaceSolidDemo", CreatePolyfaceDemo::funCreatePolyfaceSolid);
BPToolsManager::registerFun("booleanDemo", CreatePolyfaceDemo::doBoolean);
BPToolsManager::registerFun("combinePolyfaceDemo", CreatePolyfaceDemo::combinePolyface);
AutoDoRegisterFunctionsEnd