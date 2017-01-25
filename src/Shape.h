#pragma once
#ifndef _SHAPE_H_
#define _SHAPE_H_

#include <string>
#include <vector>
#include <memory>

class Program;

class Shape
{
public:
	Shape();
	virtual ~Shape();
	void loadMesh(const std::string &meshName);
	void fitToUnitBox();
	void init();
    void setOffsets(std::shared_ptr<std::vector<float>>);
	void draw(const std::shared_ptr<Program> prog) const;
    void drawInstanced(const std::shared_ptr<Program> prog, int count) const;
	
private:
	std::vector<unsigned int> eleBuf;
	std::vector<float> posBuf;
	std::vector<float> norBuf;
	std::vector<float> texBuf;
    std::vector<float> offBuf;
	unsigned eleBufID;
	unsigned posBufID;
	unsigned norBufID;
	unsigned texBufID;
    unsigned offBufID;
};

#endif
