#include "stdafx.h"
#include "DataManagerDemo.h"
#include "TgGe/gepnt3d.h"
#include "TgGe/acgetotgge.h"
#include "PBBimCore/PBTgGe.h"

#define Property_GraphicsData                          "GraphicsData"
#define Property_StringLength                          "StringLength"
#define Property_ByteLength                            "ByteLength"
#define Property_ClassName                             "ClassName"

using namespace DemoObject;
static DataManagerDemoP s_DataManager = NULL;
DataManagerDemo::DataManagerDemo() : m_byte(nullptr), m_nVersion(0)
{
	
}

DataManagerDemo::~DataManagerDemo()
{
	if (m_byte != nullptr)
		delete[] m_byte;
	
}
DataManagerDemoR DataManagerDemo::Get()
{
	if (s_DataManager == NULL)
		s_DataManager = new DataManagerDemo();
	return *s_DataManager;
}

::p3d::P3DStatus DataManagerDemo::_copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const
{
	if (T_Super::_copyToData(instance, project) != P3DStatus::SUCCESS)
		return ERROR;
	P3DStatus status;

	status = instance.setValue(Property_StringLength, BPValue(this->getStringLength()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(Property_ClassName, BPValue(this->getClassName()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	
#if 0 //CArchive方法
	BIMBase::Core::BPValue valueGraphics;

	valueGraphics.setBinary(m_byte, m_nbyteLen);
	status = instance.setValue(Property_GraphicsData, valueGraphics);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
#else  //Cereal方法
	std::vector<char> vectriFaces;
	if (!serialize_Cereal(vectriFaces))
	{
		return ERROR;
	}
	BPValue valueCereal;
	
	if (vectriFaces.size() > 0)
	{
		valueCereal.setBinary((byte*)vectriFaces.data(), vectriFaces.size());
	}
	status = instance.setValue(Property_GraphicsData, valueCereal);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	status = instance.setValue(Property_ByteLength, BPValue(int(vectriFaces.size())));
	if (P3DStatus::SUCCESS != status)
		return ERROR;
#endif
	return SUCCESS;
}

::p3d::P3DStatus DataManagerDemo::_initFromData(BIMBase::Core::BPDataCR instance)
{
	if (T_Super::_initFromData(instance) != P3DStatus::SUCCESS)
		return ERROR;
	P3DStatus status;
	BIMBase::Core::BPValue value;

	status = instance.getValue(value, Property_StringLength);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setStringLength(value.getInteger());

	status = instance.getValue(value, Property_ByteLength);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setByteLength(value.getInteger());

	status = instance.getValue(value, Property_ClassName);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setClassName(value.getWString().c_str());

	value.setBinary(0, 0);//必须初始化
	if (SUCCESS != instance.getValue(value, Property_GraphicsData))
		return ERROR;

#if 0 //CArchive方法
	size_t bytelen = 0;
	const byte* data = value.getBinary(bytelen);
	m_nbyteLen = int(bytelen);
	if (!(NULL == data || bytelen <= 0 || (int)bytelen <= 0))
	{
		m_byte = new byte[bytelen];
		for (size_t i = 0; i < bytelen; i++)
		{
			m_byte[i] = data[i];
		}
	}
	this->setByte(m_byte);
 #else //Cereal方法

	vector<char> binaryDataVec;
	size_t bytelen = 0;
	const byte* data = value.getBinary(bytelen);
	m_nByteLen = int(bytelen);
	m_byte = new byte[m_nByteLen];
	for (int i = 0; i < m_nByteLen; i++)
	{
		m_byte[i] = (char)data[i];
		binaryDataVec.push_back((char)data[i]);
	}
	this->setByte(m_byte);
	if (!deSerialize_Cereal(binaryDataVec))
		return ERROR;
#endif
	return SUCCESS;
}
BIMBase::Core::BPGraphicsPtr DemoObject::DataManagerDemo::_createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool isDynamics)
{
	BPGraphicsPtr gra = getPhysicalGraphics(project);
	return gra;
}

bool DemoObject::DataManagerDemo::_serialize_Cereal(cereal::BinaryOutputArchive& ar)const
{
	ar(m_nVersion);
	serializePoint3DVec(ar,m_pts);
	return true;
}

bool DemoObject::DataManagerDemo::_deSerialize_Cereal(cereal::BinaryInputArchive& ar)
{
	ar(m_nVersion);
	if (0 == m_nVersion)
	{
		deSerializePoint3DVec(ar,m_pts);
	}
	return true;
}


bool DemoObject::DataManagerDemo::serialize_Cereal(std::vector<char>& binaryDataVec)const
{
	//序列化
	binaryDataVec.clear();
	std::stringstream oStringstream;
	cereal::BinaryOutputArchive archiveTmp(oStringstream);
	if (_serialize_Cereal(archiveTmp))
	{
		string strTmp = oStringstream.str();
		binaryDataVec.assign(strTmp.begin(), strTmp.end());
		return true;
	}
	else
	{
		return false;
	}
	return true;
}

bool DemoObject::DataManagerDemo::deSerialize_Cereal(std::vector<char>& binaryDataVec)
{
	//反序列化
	if (binaryDataVec.size() == 0)
		return true;
	std::stringstream iStreamBuf(string(binaryDataVec.data(), binaryDataVec.size()));
	cereal::BinaryInputArchive archiveInputTmp(iStreamBuf);
	
	return _deSerialize_Cereal(archiveInputTmp);

}

bool DemoObject::DataManagerDemo::addPhysicalGraphics(BIMBase::Core::BPProjectR project, DemoObject::BaseDataDemoP Basedata)
{
	BPGraphicsPtr ptrGra = Basedata->createGraphics();
	if (ptrGra == nullptr)
		return false;
	m_classname = Basedata->getClassName();
	
#if 1	//cereal方法
	for (BPGraphics::EntryPtr& loadedEntry : *ptrGra)
	{
		m_nStrLen = 20;//只是为了兼容CArchive里需要用到这个值，cereal里不用这个值，就随便赋一个值
		BPGraphics::Entry::Type type = loadedEntry->getType();
		switch (loadedEntry->getType())
		{
		case:: BPGraphics::Entry::Type::GeCurveBase :
			IGeCurveBasePtr ptrCurveBase = loadedEntry->getAsGeCurveBaseP();
			IGeCurveBase::CurveBaseType type = ptrCurveBase->getCurveBaseType();
			switch (ptrCurveBase->getCurveBaseType())
			{
			case IGeCurveBase::CURVE_BASE_TYPE_Segment:
			{
				GeSegment3dCP segment = ptrCurveBase->getSegmentCP();
				GePoint3d start, end;
				segment->getStartAndEndPoints(start, end);
				m_pts.clear();
				m_pts.push_back(start);
				m_pts.push_back(end);
			}

			}
			break;
		}
	}
#else //CArchive方法
	
	CMemFile memFile1;
	memFile1.SeekToBegin();
	CArchive ar1(&memFile1, CArchive::store);
	ar1 << type;
	ar1.Close();
	size_t nLen1 = memFile1.GetLength();

	CMemFile memFile;
	memFile.SeekToBegin();
	CArchive ar(&memFile, CArchive::store);
	ar << type;
	PBTgGe::SerializePhysicalElement(gra, ar);
	ar.Close();
	size_t nLen = memFile.GetLength();
	if (nLen > 0)
	{
		memFile.SeekToBegin();
		CArchive arLoad(&memFile, CArchive::load);
		if (m_byte != NULL)
		{
			delete[] m_byte;
			m_byte = nullptr;

		}

		m_nbyteLen = nLen;
		m_nstrLen = nLen1;
		m_byte = new byte[m_nbyteLen];
		for (size_t i = 0; i < m_nbyteLen; i++)
		{
			arLoad >> m_byte[i];
		}

	}
#endif
	return false;

}
BIMBase::Core::BPGraphicsPtr DemoObject::DataManagerDemo::getPhysicalGraphics(BIMBase::Core::BPProjectR project)
{
	PModelId modelId = project.getDefaultModelId();
	BIMBase::Core::BPModelP pModel = project.loadModelById(modelId);
	if (pModel == NULL)
		return nullptr;
	BPGraphicsPtr tempGraphics = pModel->createPhysicalGraphics();
	GeTransform trans = getPlacement().toTransform();
	BPGraphicsUtils::transformPhysicalGraphics(*tempGraphics, trans);
	if (!tempGraphics.isValid())
		return nullptr;

#if 1	//Cereal方法
	int size = m_pts.size();
	if (size > 0)
	{
		GeSegment3d seg = GeSegment3d::create(m_pts[0], m_pts[size -1]);
		IGeCurveBasePtr curve = IGeCurveBase::createSegment(seg);
		tempGraphics->addGeCurve(*curve);
		tempGraphics->finish();
	}
	
#else	//CArchive方法

	CMemFile memFile;
	memFile.SeekToBegin();
	CArchive ar(&memFile, CArchive::store);
	CString name;
	
	for (int i =  m_nstrLen  ; i < m_nbyteLen; i++)
	{
		ar << m_byte[i];
	}
	ar.Close();
	memFile.SeekToBegin();
	CArchive arLoad(&memFile, CArchive::load);
	
	PBTgGe::SerializePhysicalElement(tempGraphics, arLoad);
	
	tempGraphics->finish();
	
	CMemFile memFile1;
	memFile1.SeekToBegin();
	CArchive ar1(&memFile1, CArchive::store);
	

	for (int i = 0; i < m_nstrLen; i++)
	{
		ar1 << m_byte[i];
	}
	ar1.Close();
	memFile1.SeekToBegin();
	CArchive arLoad1(&memFile1, CArchive::load);
	arLoad1 >> name;
#endif
	return tempGraphics;
}

pvector<byte*> DemoObject::DataManagerDemo::getAllByte()
{
	pvector<byte*> bytes;
	BPProjectP project = BIMBase::Core::BPApplication::getInstance().getProjectManager()->getMainProject();
	if (project == NULL)
		return bytes;
	BPDataClass ecClass;
	p3d::Utf8String wSchemaName = this->_getSchemaName().data();
	p3d::Utf8String wClassName = this->_getClassName().data();
	if (P3DStatus::SUCCESS != BPSchemaManager::getClassByName(ecClass, wSchemaName, wClassName, *project))
		return bytes;
	BPDataList instanceList;
	P3DStatus status = BPDataUtil::getDataFromSchemaName(instanceList, PBM_SCHEMA_Demo, PBM_CLASS_DATAMANAGER_Demo, *project);
	if(status != 0 || instanceList.empty())
		return bytes;
	for (auto instance : instanceList) 
	{
		if (!instance.isValid()) continue;
		DataManagerDemoP ptrdata = DataManagerDemo::create(*instance);
		if (ptrdata == NULL)
			continue;
		int nStrsize = ptrdata->getStringLength();
		m_nStrvec.push_back(nStrsize);
		int bytesize = ptrdata->getByteLength();
		m_nBytevec.push_back(bytesize);
		byte* byt = ptrdata->getByte();
		m_classname = ptrdata->getClassName();
		bytes.push_back(byt);
	}
	return bytes;
}
