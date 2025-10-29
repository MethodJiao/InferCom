#include "stdafx.h"
#include "ObjectChangeEventDemo.h"

ObjectChangeEventDemo::ObjectChangeEventDemo()
{
}


ObjectChangeEventDemo::~ObjectChangeEventDemo()
{
}

bool ObjectChangeEventDemo::_onPostNew(IBPObjectCR arg)
{
	
	return false;
}


bool ObjectChangeEventDemo::_onPostEdit(IBPObjectCR arg)
{

	return false;
}

bool ObjectChangeEventDemo::_onPreDelete(IBPObjectCR pbObject)
{
	//·µ»Øtrue¿É×èÖ¹É¾³ı
	return false;
}
