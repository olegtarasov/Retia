namespace Retia.Training.Trainers
{
    public interface ITrainingStatusWriter
    {
        void UpdateItemStatus(string status);
        void ItemComplete();
        void Message(string message);
    }
}