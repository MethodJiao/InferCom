#include "stdafx.h"

using namespace p3d;
USING_NAMESPACE_TRACER

TgGePoint3d ChangePoint3d(IN GePoint3d src)
{
	return TgGePoint3d(src.x, src.y, src.z);
}

GeVec3d ChangeVector3d(IN TgGeVector3d src)
{
	GeVec3d vt;
	vt.set(src.x, src.y, src.z);
	return vt;
}

TgGeVector3d ChangeVector3d(IN GeVec3d src)
{
	return TgGeVector3d(src.x, src.y, src.z);
}


TracerCusFunDemo::TracerCusFunDemo()
{
}


TracerCusFunDemo::~TracerCusFunDemo()
{
}

// 设置初始化时追踪器面板格式
void TracerCusFunDemo::_OnPlaceCtrl(OUT Tracer::CtrlList& ref)
{
	Tracer::Item sweepItem(L"长度", 2, false, true);
	ref.setCustomItem(L"长度", sweepItem);
	ref.setDisAngItem();
	ref.setHeightItem(2, true, true);
	ref.setXItem();
	ref.setYItem();
	ref.setZItem(2, true, true);
}

void TracerCusFunDemo::_OnUpdateUi(OUT Tracer::CtrlList& ref, GePoint3dCR ptLast, GePoint3dCR ptNew)
{
	// 坐标原点
	GePoint3d ptTempOri = GePoint3d::createByZero();
	BPTempUcsMngr::getOrigin(ptTempOri);
	// 计算距离
	double dDis = ptTempOri.distance(ptNew);

	//获取动态UCS法向
	GeRotMatrix matTU = GeRotMatrix::createIdentityMatrix();
	BPTempUcsMngr::getRotation(matTU);
	GeVec3d vtXTmp = GeVec3d::create(1, 0, 0), vtYTmp = GeVec3d::create(0, 1, 0),
		vtZTmp = GeVec3d::create(0, 0, 1);
	matTU.getRows(vtXTmp, vtYTmp, vtZTmp);
	// 计算角度
	GePoint3d ptPoint = ptNew;
	GePlane3d workPlane;
	workPlane.origin = ptTempOri;
	workPlane.normal = vtZTmp;
	GeRay3d ray;
	ray.origin = ptPoint;
	ray.direction = vtZTmp;
	double dPara;
	ray.getIntersectPointWithPlane(ptPoint, dPara, workPlane);
	double dAngle = 0.00;
	if (!ptPoint.almostEqual(ptTempOri))
	{
		GeVec3d vtX = ptPoint - ptTempOri;
		dAngle = vtXTmp.signedAngleTo(vtX, vtZTmp);
		dAngle = (dAngle * 180) / PI;
	}
	if (fabs(dAngle) < 1e-6)
	{
		dAngle = 0.0;
	}
	// 获取相对坐标的值
	GePoint3d   ptTemp = ptNew - ptTempOri;
	GeTransform tempTrans = GeTransform::create(matTU);
	ptTemp = GePoint3d::createByTransform(tempTrans, ptTemp);
	// 根据坐标显示状态 转换XYZ显示参数
	double dX, dY, dZ;
	if (ref.m_bLocal)
	{
		dX = ptTemp.x;
		dY = ptTemp.y;
		dZ = ptTemp.z;
	}
	else
	{
		dX = ptNew.x;
		dY = ptNew.y;
		dZ = ptNew.z;
	}
	double dH = ptTemp.z;
	if (fabs(dH) < 1e-6)
	{
		dH = 0.0;
	}
	// 更新计算内容
	ref.setCustomValue(L"长度", std::fabs(dDis));
	ref.setDisValue(dDis);
	ref.setAngValue(dAngle);
	ref.setHeightValue(dH);
	ref.setXValue(dX);
	ref.setYValue(dY);
	ref.setZValue(dZ);
}

bool TracerCusFunDemo::_OnSurePlace(OUT Tracer::CtrlList& ref, IN GePoint3dCR ptLast, OUT GePoint3d& ptNew)
{ 
	// 坐标原点
	GePoint3d ptTempOri = GePoint3d::createByZero();
	BPTempUcsMngr::getOrigin(ptTempOri);
	// 0,长度 1, 距离 2 角度 3 高度 4 X坐标 5 Y坐标 6 Z坐标
	if (ref.m_nlastSelId == 0 || ref.m_nlastSelId == 1 || ref.m_nlastSelId == 2 || ref.m_nlastSelId == 3)
	{
		//长度
		double dLength = fabs(ref.getCustomValue(L"长度"));
		// 距离
		double dDis = fabs(ref.getDisValue());
		if (fabs(dLength - dDis) > 1e-6)
		{
			dDis = dLength;
		}
		// 角度
		double dAng = ref.getAngValue();
		// 高度
		double dHeight = ref.getHeightValue();
		if (dDis < 1e-8)
		{
			dAng = 0;
			dHeight = 0;
		}
		if (dDis - fabs(dHeight) < -1e-6)
		{
			PBMessageCenter::Send(P3DApi::P3DApplication::JsonMessage(
				PBBim_MESSAGE_ToolPrompt, Utf8String(L"极坐标距离参数不能小于高度参数").c_str()));
			return false;
		}
		// 计算投影线长度
		double dProjectionDis = sqrt(abs(pow(dDis, 2) - pow(abs(dHeight), 2)));
		//获取动态UCS法向
		GeRotMatrix matTU;
		BPTempUcsMngr::getRotation(matTU);
		GeVec3d vtXTmp, vtYTmp, vtZTmp;
		matTU.getRows(vtXTmp, vtYTmp, vtZTmp);
		vtXTmp.normalize();
		vtXTmp *= dProjectionDis;
		double    dTheta = dAng / 180 * PI;
		GeAngle   angle = GeAngle::createByDegrees(dTheta);
		GePoint3d newPt = ptTempOri;
		newPt += vtXTmp;
		TgGePoint3d  pt = ChangePoint3d(newPt);
		TgGeVector3d vtZTemp = ChangeVector3d(vtZTmp);
		TgGePoint3d  ptOri = ChangePoint3d(ptTempOri);
		pt.rotateBy(dTheta, vtZTemp, ptOri);
		ptNew.x = pt.x;
		ptNew.y = pt.y;
		ptNew.z = pt.z;
		// 计算实际空间点
		vtZTmp *= dHeight;
		ptNew += vtZTmp;
	}
	else
	{
		// 根据输入的XYZ计算ptNew
		if (ref.m_bLocal)
		{
			GePoint3d ptTemp = GePoint3d::create(ref.getXValue(), ref.getYValue(), ref.getZValue());
			// 将显示的坐标转换为世界坐标系的值
			GeRotMatrix matTU;
			BPTempUcsMngr::getRotation(matTU);
			matTU.inverse();
			GeTransform tempTrans = GeTransform::create(matTU);
			ptTemp = GePoint3d::createByTransform(tempTrans, ptTemp);
			ptNew.x = ptTemp.x + ptTempOri.x;
			ptNew.y = ptTemp.y + ptTempOri.y;
			ptNew.z = ptTemp.z + ptTempOri.z;
		}
		else
		{
			ptNew.x = ref.getXValue();
			ptNew.y = ref.getYValue();
			ptNew.z = ref.getZValue();
		}
	}
}

static TracerCusFunDemo s_TracerFun;
AutoDoRegisterFunctionsBegin
Tracer::REG_TRACER("Tracer.SecDevCustom", &s_TracerFun);
AutoDoRegisterFunctionsEnd