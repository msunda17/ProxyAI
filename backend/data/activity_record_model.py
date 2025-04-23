from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from data.choice_data_enums import (
    SustainableDevelopmentGoal,
    TargetPopulation,
    FocusAreaCategory,
    GoalOutput,
    GoalInstitutionalOutcome,
    GoalCommunityImpact,
    FocusAreaCategoryArtsAndCulture,
    FocusAreaCategoryCommunityAndEconomicDevelopment,
    FocusAreaCategoryEducation,
    FocusAreaCategoryEnvironmentalSustainability,
    FocusAreaCategoryGovernmentAndPublicSafety,
    FocusAreaCategoryHealthandWellness,
    FocusAreaCategorySocialIssues,
    ActivityType
)

class Contact(BaseModel):
    firstName: str
    lastName: str
    email: str
    phone: str

class DateRange(BaseModel):
    startDate: str
    endDate: str

class Organization(BaseModel):
    name: str
    primaryContact: Contact

class FundingSource(BaseModel):
    funderName: str
    isInternalSource: bool
    amount: float
    startDate: str
    endDate: str

class FundingDetails(BaseModel):
    isFunded: bool
    funders: List[FundingSource]

class Address(BaseModel):
    country: str
    street1: str
    street2: str
    zipCode: str

class Location(BaseModel):
    isPhysical: bool
    physicalSites: List[Address]

class CourseDetail(BaseModel):
    isConnected: bool
    courseName: str
    studentEnrollCount: int
    isEnrollCountActual: bool
    studentContributingHours: int
    isContributingHoursActual: bool
    pedagogies: List[str]  # You can replace with a Pedagogy Enum
    studentLearningObjectives: List[str]  # You can replace with a LearningObjective Enum

class ResearchDetail(BaseModel):
    isConnected: bool
    researchType: str
    scholarlyProducts: List[str]

class GoalSection(BaseModel):
    expected: List[GoalOutput]
    achieved: List[GoalOutput]

class InstitutionalOutcomeSection(BaseModel):
    expected: List[GoalInstitutionalOutcome]
    achieved: List[GoalInstitutionalOutcome]

class CommunityImpactSection(BaseModel):
    expected: List[GoalCommunityImpact]
    achieved: List[GoalCommunityImpact]

class Goals(BaseModel):
    outputs: GoalSection
    institutionalOutcomes: InstitutionalOutcomeSection
    communityImpacts: CommunityImpactSection

class DataCollectionDetails(BaseModel):
    isSystematic: bool
    description: str

class FocusAreasWithinCategories(BaseModel):
    artsAndCulture: List[FocusAreaCategoryArtsAndCulture]
    communityAndEconomicDevelopment: List[FocusAreaCategoryCommunityAndEconomicDevelopment]
    education: List[FocusAreaCategoryEducation]
    environmentalSustainability: List[FocusAreaCategoryEnvironmentalSustainability]
    governmentAndPublicSafety: List[FocusAreaCategoryGovernmentAndPublicSafety]
    healthAndWellness: List[FocusAreaCategoryHealthandWellness]
    socialIssues: List[FocusAreaCategorySocialIssues]

class ActivityType(BaseModel):
    activityType: List[ActivityType]

class ActivityRecord(BaseModel):
    activityType: str
    activityTitle: str
    activityDescription: str
    activityWebsite: str
    primaryContact: Contact
    activityDates: DateRange
    units: List[str]
    programsOrInitiatives: List[SustainableDevelopmentGoal]
    facultyOrStaff: List[str]
    participantFacultyStaff: int
    communityOrganizations: List[Organization]
    otherInstitutions: dict  # You can model with nested Organization list for 'k12schools' and 'institutions'
    fundingDetails: FundingDetails
    activityLocationDetails: Location
    targetPopulation: List[TargetPopulation]
    focusAreaCategories: List[FocusAreaCategory]
    focusAreasWithinCategories: FocusAreasWithinCategories
    coursesDetails: List[CourseDetail]
    researchDetails: List[ResearchDetail]
    goals: Goals
    communityIndividualsServed: int
    communityViewsDescription: str
    dataCollectionDetails: DataCollectionDetails