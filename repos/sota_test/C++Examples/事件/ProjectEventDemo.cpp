#include "stdafx.h"
#include "ProjectEventDemo.h"
#include "MyCubeDragManipulatorDemo.h"
#include "CubeDemo.h"
#include "ProjectEventDemo.h"
#include "DomainChangeEventDemo.h"



bool ProjectEventDemo::_onPreOpen(const ProjectPreOpenArg& arg)
{
	return false;
}


bool ProjectEventDemo::_onPostOpen(BIMBase::Core::BPProjectR project)
{

	BIMBase::Data::BPDomainEventHandleManager::getInstance()->refreshAll(BPDomainEnvironment::getInstance()->getDomainCodeByKeyName(L"二次开发CPP"));
	p3d::Utf8String str = project.getGuid();
	int a = 0;
	return false;

}

bool ProjectEventDemo::_onPreClose(BIMBase::Core::BPProjectR project)
{
	return false;
}

bool ProjectEventDemo::_onPostClose(BIMBase::Core::BPProjectR project)
{
	return false;
}

