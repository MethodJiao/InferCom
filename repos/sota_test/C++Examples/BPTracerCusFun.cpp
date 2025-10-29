#include "stdafx.h"


using namespace p3d;

BPTracerCusFun::BPTracerCusFun()
{
}


BPTracerCusFun::~BPTracerCusFun()
{
}

void BPTracerCusFun::_OnUpdateUi(wd::CtrlList& ref, GePoint3dCR ptLast, GePoint3dCR ptNew)
{
	switch (ref.type)
	{
	case eCusTracerType::DISTANCE:
	{
		double dDis = ptNew.distance(ptLast);
		double dAng = (ptNew - ptLast).signedAngleTo(GeVec3d::create(1, 0, 0), GeVec3d::create(0, 0, -1));
		dAng = dAng*180/PI;
		vector<double> vctValue;
		vctValue.push_back(dDis);
		vctValue.push_back(dAng);
		vctValue.push_back(ptNew.x);
		vctValue.push_back(ptNew.y);
		vctValue.push_back(ptNew.z);
		CString csTmp;
		for (auto& item : ref.m_mapIdx2Cs)
		{
			csTmp.Format(L"%.2f", vctValue[item.first]);
			item.second = csTmp;
		}	
	}
	break;
	case eCusTracerType::WORLD:
	{
		vector<double> vctValue;;
		vctValue.push_back(ptNew.x);
		vctValue.push_back(ptNew.y);
		vctValue.push_back(ptNew.z);
		CString csTmp;
		for (auto& item : ref.m_mapIdx2Cs)
		{
			csTmp.Format(L"%.2f", vctValue[item.first]);
			item.second = csTmp;
		}
	}
	default:
		break;
	}
}

void BPTracerCusFun::_OnPlaceCtrl(wd::CtrlList& ref)
{
	switch (ref.type)
	{
	case eCusTracerType::WORLD:
	{
		map<UINT, CString> tmp = { { 0, L"X坐标" }, { 1, L"Y坐标" }, { 2, L"Z坐标" } };
		ref.m_mapIdx2Cs = tmp;
	}
	break;
	case eCusTracerType::DISTANCE:
	{
		map<UINT, CString> tmp = { { 0, L"距离" }, { 1, L"角度" }, { 2, L"X坐标" }, { 3, L"Y坐标" }, { 4, L"Z坐标" }};
		ref.m_mapIdx2Cs = tmp;
	}
	break;
	default:
		break;
	}
}

void BPTracerCusFun::_OnSurePlace(wd::CtrlList& ref, GePoint3dCR ptLast, GePoint3d& ptNew)
{
	switch (ref.type)
	{
	case eCusTracerType::WORLD:
	{
		if (ref.m_mapIdx2Cs.size() < 3)
			break;
		ptNew.x = _ttof(ref.m_mapIdx2Cs[0]);
		ptNew.y = _ttof(ref.m_mapIdx2Cs[1]);
		ptNew.z = _ttof(ref.m_mapIdx2Cs[2]);
		_OnUpdateUi(ref, ptLast, ptNew);
	}
	break;
	case eCusTracerType::DISTANCE:
	{
		if (ref.m_mapIdx2Cs.size() < 5)
		break;
		if (ref.index == 0 || ref.index == 1)
		{
			CString csDis = ref.m_mapIdx2Cs[0];
			CString csAng = ref.m_mapIdx2Cs[1];
			double dDis = _ttof(csDis);
			double dAng = _ttof(csAng);
			dAng = dAng*PI/180;
			GeVec3d vecX = GeVec3d::create(1, 0, 0);
			vecX *= dDis;
			vecX.rotate2D(dAng);
			GePoint3d newPt = ptLast + vecX;			
			ptNew = newPt;
			_OnUpdateUi(ref, ptLast, ptNew);
		}
		else
		{
			ptNew.x = _ttof(ref.m_mapIdx2Cs[2]);
			ptNew.y = _ttof(ref.m_mapIdx2Cs[3]);
			ptNew.z = _ttof(ref.m_mapIdx2Cs[4]);
			_OnUpdateUi(ref, ptLast, ptNew);
		}
	}
	break;
	default:
		break;
	}
}


static BPTracerCusFun s_TracerFun;
AutoDoRegisterFunctionsBegin
wd::REG_TRACER_FUNPOOL("BPTracer.Custom", &s_TracerFun);
AutoDoRegisterFunctionsEnd